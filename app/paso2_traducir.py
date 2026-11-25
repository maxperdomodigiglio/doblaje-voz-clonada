"""Paso 2 (automatico, sin API key): corrige la transcripcion en castellano y traduce con un modelo local.

Para cada capitulo preparado en el Paso 1:
  1. corrige la transcripcion (glosario fijo + modelo local, solo cambios chicos y verificados)
     -> es_corregido.json y es.srt (subtitulos en castellano, si no habia unos revisados)
  2. arma los bloques de doblaje y traduce cada uno al idioma elegido (ingles, aleman o ruso)
     -> en.json / de.json / ru.json  (marcados "auto": true)
  3. deja revisar_XX.txt con lo dudoso, para curarlo despues con Claude.

Nunca pisa una traduccion revisada a mano (las que no tienen "auto": true o tienen "curado": true).
Con --forzar rehace las automaticas.
"""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

from comun import (APP_DIR, ORIGENES, idioma_origen, progreso, archivo_traduccion, carpeta_proceso, ejecutar_principal, escribir_json, escribir_srt,
                   leer_json, log, log_error, normalizar_palabras)
from textos import bloques_por_oraciones, corregir_segmentos, cues_desde_palabras, palabras_limpias

FORZAR = "--forzar" in sys.argv
SIMULAR = "--simular" in sys.argv  # pruebas sin GPU ni modelo: no corrige y "traduce" marcando el texto
SOLO = [a for a in sys.argv[1:] if not a.startswith("--")]
VERSION_CORR = 1
CONTEXTO = "Una persona habla a camara o en un audio, en primera persona."


# ---------------------------------------------------------------- modelo local
class LLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tok = self.model = None

    def cargar(self):
        if self.model is not None or SIMULAR:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        nombre = self.cfg.get("modelo_traduccion", "Qwen/Qwen3-4B-Instruct-2507")
        cache = str(APP_DIR / "modelos" / "hf")
        log(f"  Cargando el modelo de traduccion {nombre} (la primera vez se descarga, unos 8 GB)...")
        self.tok = AutoTokenizer.from_pretrained(nombre, cache_dir=cache)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            nombre, cache_dir=cache, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map=dev)
        self.model.eval()

    def chat(self, sistema, usuario, max_new_tokens=700, temperatura=0.0):
        if SIMULAR:
            return _respuesta_simulada(usuario)
        import torch
        self.cargar()
        msgs = [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        gen = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.05, pad_token_id=self.tok.eos_token_id)
        if temperatura > 0:
            gen.update(do_sample=True, temperature=temperatura, top_p=0.9)
        else:
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **gen)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def _respuesta_simulada(usuario):
    if "<<<LINEAS>>>" in usuario:  # correccion: devuelve las mismas lineas
        return usuario.split("<<<LINEAS>>>", 1)[1].strip()
    return "[TRAD] " + usuario.rsplit("<<<FRAGMENTO>>>", 1)[1].strip()


# ---------------------------------------------------------------- glosario
def cargar_glosario():
    p = APP_DIR / "glosario.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aplicar_glosario_es(texto, glos):
    for patron, reemplazo in glos.get("correcciones_es", []):
        texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)
    return texto


# ---------------------------------------------------------------- 1) correccion del castellano
SIS_ES = (
    "Sos corrector de transcripciones automaticas (Whisper) de audios en {idioma}. {contexto}\n"
    "Te paso lineas numeradas. Corregi SOLO:\n"
    "- puntuacion, mayusculas y signos de pregunta/exclamacion de apertura;\n"
    "- errores evidentes de transcripcion: nombres propios mal escritos (musicos, artistas, autores, bandas) y "
    "palabras que no tienen sentido por la parecida que si lo tiene.\n"
    "No resumas, no cambies el estilo ni el orden, no agregues ni saques ideas, no unas ni separes lineas. "
    "Si una linea esta bien, devolvela igual.\n"
    "Respuesta en el mismo idioma de las lineas: exactamente las mismas lineas, una por renglon, con el formato "
    "'N: texto'. Nada mas."
)


def _valida(orig, nuevo):
    a, b = normalizar_palabras(orig), normalizar_palabras(nuevo)
    if not b:
        return False
    if abs(len(a) - len(b)) > max(2, int(0.2 * len(a))):
        return False
    return difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio() >= 0.75


def corregir_castellano(es, llm, glos, avance=None, src="es"):
    segs = es["segmentos"]
    textos = [aplicar_glosario_es(sg["text"], glos) if src == "es" else sg["text"] for sg in segs]
    final = list(textos)
    cambios, rechazados = [], []
    nombres = ", ".join(glos.get("nombres", []))
    tanda = int(llm.cfg.get("correccion_tanda", 10))
    for i in range(0, len(segs), tanda):
        if avance:
            avance(i, len(segs))
        grupo = textos[i:i + tanda]
        ctx = "\n".join(final[max(0, i - 3):i])
        usuario = (f"Nombres que aparecen en la serie: {nombres}.\n\n"
                   + (f"Lo anterior (ya corregido, solo de contexto):\n{ctx}\n\n" if ctx else "")
                   + "Lineas a corregir:\n<<<LINEAS>>>\n"
                   + "\n".join(f"{k + 1}: {t}" for k, t in enumerate(grupo)))
        try:
            resp = llm.chat(SIS_ES.format(contexto=llm.cfg.get("contexto") or CONTEXTO,
                                          idioma=ORIGENES.get(src, (src, src))[1]), usuario, max_new_tokens=60 + 2 * sum(len(t) for t in grupo) // 3)
        except Exception:
            log_error("correccion del castellano (sigo con la transcripcion original)")
            continue
        for linea in resp.splitlines():
            m = re.match(r"\s*(\d+)\s*[:.)-]\s*(.*)$", linea)
            if not m:
                continue
            k = int(m.group(1)) - 1
            if not 0 <= k < len(grupo):
                continue
            nuevo = m.group(2).strip()
            if nuevo == grupo[k]:
                continue
            if _valida(grupo[k], nuevo):
                final[i + k] = nuevo
                cambios.append((segs[i + k]["start"], grupo[k], nuevo))
            else:
                rechazados.append((segs[i + k]["start"], grupo[k], nuevo))
    correcciones = {sg["start"]: t for sg, t in zip(segs, final) if t != sg["text"]}
    corregidos = corregir_segmentos(segs, correcciones)
    for k, sg in enumerate(corregidos):
        sg["id"] = k + 1
    return corregidos, cambios, rechazados


# ---------------------------------------------------------------- 2) traduccion
def sistema_traduccion(cfg, src="es"):
    lang = cfg["idioma_destino"]
    origen = ORIGENES.get(src, (src,))[0]
    extra = {
        "ru": " In Russian, write people's and bands' names in Cyrillic as usual (e.g. Курт Кобейн), "
              "but keep song and album titles in their original language.",
        "de": " Use natural spoken German (du/Sie as fits a talk to the audience: address viewers with 'Sie').",
        "es": " Use neutral Latin American Spanish.",
    }.get(cfg["idioma_codigo"], "")
    return (
        f"You are an expert audiovisual translator preparing a voice-over dub from {origen} into natural, fluent "
        f"spoken {lang}. About the video (written in Spanish): {cfg.get('contexto') or CONTEXTO}\n"
        f"Rules:\n"
        f"- Translate the fragment completely and faithfully, in first person, the way the speaker would say it in {lang}. "
        f"No explanations, no notes, no quotation marks around the whole text.\n"
        f"- The fragment may start or end mid-sentence: keep it that way so it joins naturally with the next one.\n"
        f"- It is read aloud over the same time slot: keep it about as long as the original, prefer concise wording.\n"
        f"- Keep names of people, bands and songs as they are usually written in {lang}.{extra}\n"
        f"- Write numbers as digits.\n"
        f"- Output only the {lang} translation."
    )


_ES_FRECUENTES = {"que", "el", "los", "las", "por", "una", "pero", "porque", "como", "esto", "muy", "también", "tambien"}


def _problema_traduccion(es_txt, tr, cod, src="es"):
    if not tr:
        return "vacia"
    r = len(tr) / max(1, len(es_txt))
    if r < 0.4 or r > 2.0:
        return f"largo raro ({r:.1f}x el original)"
    pal = normalizar_palabras(tr)
    if cod == "ru" and not re.search(r"[А-Яа-яЁё]", tr):
        return "no parece ruso"
    if src == "es" and cod != "es" and pal and sum(p in _ES_FRECUENTES for p in pal) / len(pal) > 0.15:
        return "quedo castellano sin traducir"
    return None


def _limpiar(tr, cfg):
    tr = tr.strip()
    tr = re.sub(rf"^\s*(?:{cfg['idioma_destino']}|translation|traducci[oó]n|übersetzung|перевод)\s*:\s*", "", tr,
                flags=re.IGNORECASE)
    if len(tr) > 2 and tr[0] == tr[-1] and tr[0] in "\"'«“":
        tr = tr[1:-1].strip()
    tr = tr.replace("“", "\"").replace("”", "\"")
    return " ".join(tr.split())


def traducir_bloques(bloques, llm, cfg, ref_en=None, avance=None, parcial=None, src="es"):
    """parcial: archivo donde se guarda cada bloque traducido, para retomar si se corta o se detiene."""
    sis = sistema_traduccion(cfg, src)
    L = cfg["idioma_destino"]
    hechos, avisos = [], []
    cache = {}
    if parcial and parcial.exists():
        try:
            cache = json.loads(parcial.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    for i, (s, e, txt) in enumerate(bloques):
        if avance:
            avance(i, len(bloques))
        clave = hashlib.md5(f"{cfg['idioma_codigo']}|{txt}|{ref_en[i] if ref_en else ''}".encode()).hexdigest()[:16]
        if clave in cache:
            hechos.append({"es": txt, "tr": cache[clave]})
            continue
        previos = hechos[-2:]
        ctx = "".join(f"ES: {b['es']}\n{L}: {b['tr']}\n" for b in previos)
        sig = bloques[i + 1][2] if i + 1 < len(bloques) else ""
        usuario = ((f"Previous fragments (context only, already translated):\n{ctx}\n" if ctx else "")
                   + (f"Next fragment (context only, do not translate): {sig}\n\n" if sig else "")
                   + (f"Approved English translation of this same fragment (use it to understand names and meaning; "
                      f"translate from the Spanish): {ref_en[i]}\n\n" if ref_en and L != "English" else "")
                   + f"Translate this fragment into {L}:\n<<<FRAGMENTO>>>\n{txt}")
        tr, problema = "", "vacia"
        for intento, temp in enumerate((0.0, 0.4)):
            try:
                tr = _limpiar(llm.chat(sis, usuario, max_new_tokens=80 + len(txt), temperatura=temp), cfg)
            except Exception:
                log_error(f"traduccion del bloque {i + 1}")
                tr = ""
            problema = _problema_traduccion(txt, tr, cfg["idioma_codigo"], src) if not SIMULAR else None
            if not problema:
                break
        if problema:
            avisos.append((i + 1, s, problema, txt, tr))
        hechos.append({"es": txt, "tr": tr or txt})
        if parcial and tr and not problema:
            cache[clave] = tr
            parcial.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        log(f"  bloque {i + 1:3d}/{len(bloques)}  {tr[:70]}")
    return [b["tr"] for b in hechos], avisos


# ---------------------------------------------------------------- texto para la voz (numeros en palabras)
def texto_para_voz(txt, cod, glos):
    for palabra, como in glos.get("pronunciacion", {}).get(cod, {}).items():
        txt = re.sub(rf"\b{re.escape(palabra)}\b", como, txt)
    try:
        from num2words import num2words
    except ImportError:
        return txt

    def num(m):
        s = m.group(0)
        try:
            if re.fullmatch(r"\d+", s):
                n = int(s)
                if 1100 <= n <= 2099 and cod in ("en", "de"):
                    try:
                        return num2words(n, lang=cod, to="year")
                    except Exception:
                        pass
                return num2words(n, lang=cod)
            return num2words(float(s.replace(",", ".")), lang=cod)
        except Exception:
            return s

    txt = re.sub(r"\d+(?:[.,]\d+)?", num, txt)
    pct = {"en": " percent", "de": " Prozent", "ru": " процентов", "es": " por ciento"}.get(cod, "%")
    return txt.replace("%", pct)


# ---------------------------------------------------------------- por capitulo
def es_corregido(d, es, llm, glos, avance=None, src="es"):
    p = d / "es_corregido.json"
    if p.exists() and not FORZAR:
        prev = leer_json(p)
        if prev.get("version") == VERSION_CORR and p.stat().st_mtime >= (d / "es.json").stat().st_mtime:
            return prev
    log(f"[{d.name}] Corrigiendo la transcripcion ({ORIGENES.get(src, (src, src))[1]})...")
    segs, cambios, rechazados = corregir_castellano(es, llm, glos, avance, src)
    data = {"version": VERSION_CORR, "segmentos": segs,
            "cambios": [{"t": round(t, 2), "antes": a, "despues": b} for t, a, b in cambios],
            "rechazados": [{"t": round(t, 2), "antes": a, "propuesta": b} for t, a, b in rechazados]}
    escribir_json(p, data)
    log(f"  {len(cambios)} correcciones aplicadas, {len(rechazados)} descartadas por cambiar demasiado")
    return data


def traducir_capitulo(d, cfg, llm, glos):
    es = leer_json(d / "es.json")
    src = idioma_origen(es, cfg)
    destino = archivo_traduccion(d, cfg)
    if destino.exists():
        prev = leer_json(destino)
        if not prev.get("auto") or prev.get("curado"):
            log(f"[{d.name}] traduccion al {cfg['idioma_nombre']} ya revisada, no la toco")
            progreso(d.name, 2, 100, "traducción ya revisada")
            return
        if not FORZAR:
            log(f"[{d.name}] ya traducido al {cfg['idioma_nombre']} (automatico), salteo")
            progreso(d.name, 2, 100, "ya estaba traducido")
            return

    progreso(d.name, 2, 1, "cargando el traductor")
    parcial = d / f"_parcial_{cfg['idioma_codigo']}.json"

    def av_corr(i, n):
        progreso(d.name, 2, 2 + 28 * i / max(n, 1), "corrigiendo la transcripción")

    def av_trad(i, n, base=30):
        progreso(d.name, 2, base + (98 - base) * i / max(n, 1), f"traduciendo frase {i + 1} de {n}")

    # si ya hay una version en ingles revisada, se usan sus mismos bloques y sirve de referencia
    en_rev = None
    if cfg["idioma_codigo"] not in ("en", "es") and src == "es" and (d / "en.json").exists():
        en_prev = leer_json(d / "en.json")
        if not en_prev.get("auto") or en_prev.get("curado"):
            en_rev = en_prev["bloques"]
    if en_rev:
        corr = {}
        bloques = [(b["start"], b["end"], b["es"]) for b in en_rev]
        log(f"[{d.name}] Traduciendo {len(bloques)} bloques al {cfg['idioma_nombre']} "
            f"(con la version en ingles revisada de referencia)...")
        trads, avisos = traducir_bloques(bloques, llm, cfg, ref_en=[b["en"] for b in en_rev],
                                         avance=lambda i, n: av_trad(i, n, base=3), parcial=parcial, src=src)
    else:
        corr = es_corregido(d, es, llm, glos, av_corr, src)
        segs = corr["segmentos"]
        if not (d / "es.srt").exists():
            cues = cues_desde_palabras(segs)
            escribir_srt(d / "es.srt", [tuple(c) for c in cues])
            log(f"  Subtitulos del original: {len(cues)}")
        ws = palabras_limpias({"segmentos": segs})
        bloques = bloques_por_oraciones(ws, max_dur=float(cfg.get("bloque_max_seg", 14.0)) - 1.0)
        if src == cfg["idioma_codigo"]:
            # mismo idioma: no hay que traducir, solo se cambia la voz
            log(f"[{d.name}] El audio ya esta en {cfg['idioma_nombre']}: no traduzco, solo cambio la voz")
            trads, avisos = [b[2] for b in bloques], []
        else:
            log(f"[{d.name}] Traduciendo {len(bloques)} bloques al {cfg['idioma_nombre']}...")
            trads, avisos = traducir_bloques(bloques, llm, cfg, avance=av_trad, parcial=parcial, src=src)

    cod = cfg["idioma_codigo"]
    salida = []
    for k, ((s, e, txt), tr) in enumerate(zip(bloques, trads), 1):
        b = {"id": k, "start": round(s, 3), "end": round(e, 3), "es": txt, "en": tr}
        voz = texto_para_voz(tr, cod, glos)
        if voz != tr:
            b["tts"] = voz
        salida.append(b)
    escribir_json(destino, {"slug": d.name, "video": es["video"], "idioma": cod, "origen": src, "auto": True,
                            "modelo": cfg.get("modelo_traduccion", "Qwen/Qwen3-4B-Instruct-2507"), "bloques": salida})
    escribir_revision(d, cfg, es, corr, avisos)
    parcial.unlink(missing_ok=True)
    progreso(d.name, 2, 100, "listo")
    log(f"[{d.name}] LISTO paso 2 -> {destino.name} ({len(salida)} bloques, {len(avisos)} para revisar)")


def escribir_revision(d, cfg, es, corr, avisos):
    cod = cfg["idioma_codigo"]
    r = [f"REVISION {d.name} -> {cfg['idioma_destino']}  (para curar con Claude)", ""]
    r.append("== Traducciones dudosas ==")
    r += [f"  bloque {i} ({t:.1f}s): {p}\n    ES: {es_t}\n    {cod.upper()}: {tr}" for i, t, p, es_t, tr in avisos] or ["  ninguna"]
    r += ["", "== Palabras que Whisper no entendio bien (confianza baja) =="]
    dudosas = [(w["s"], w["w"]) for sg in es["segmentos"] for w in sg["words"] if w.get("p", 1) < 0.45]
    r += [f"  {t:7.1f}s  {w}" for t, w in dudosas] or ["  ninguna"]
    r += ["", "== Correcciones del castellano aplicadas =="]
    r += [f"  {c['t']:7.1f}s  {c['antes']}\n           -> {c['despues']}" for c in corr.get("cambios", [])] or ["  ninguna"]
    r += ["", "== Correcciones propuestas y descartadas (cambiaban demasiado) =="]
    r += [f"  {c['t']:7.1f}s  {c['antes']}\n           ?? {c['propuesta']}" for c in corr.get("rechazados", [])] or ["  ninguna"]
    (d / f"revisar_{cod}.txt").write_text("\n".join(r) + "\n", encoding="utf-8")


def main(cfg):
    proc = carpeta_proceso(cfg)
    caps = sorted(p for p in proc.iterdir() if p.is_dir() and (p / "es.json").exists())
    if SOLO:
        caps = [p for p in caps if p.name in SOLO]
    if not caps:
        log("No hay capitulos preparados (falta el Paso 1).")
        return
    log(f"Idioma: {cfg['idioma_nombre']} | modelo: {cfg.get('modelo_traduccion', 'Qwen/Qwen3-4B-Instruct-2507')}")
    llm, glos = LLM(cfg), cargar_glosario()
    errores = 0
    for d in caps:
        try:
            traducir_capitulo(d, cfg, llm, glos)
        except Exception:
            errores += 1
            log_error(d.name)
    log(f"Paso 2 terminado. Errores: {errores}")
    if errores:
        sys.exit(1)


if __name__ == "__main__":
    ejecutar_principal("PASO 2 (traducir)", main)
