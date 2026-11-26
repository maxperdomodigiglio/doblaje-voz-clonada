"""Texto para tu voz: escribis (o cargas) un texto y la app lo dice con la voz clonada de la muestra.

  python decir.py --config=<ajustes_texto.json>

En el json: carpeta_videos, voz_muestra, texto_archivo (un .txt con el texto) y texto_idioma
(es/en/de/ru/fr/it/pt o "auto"). Lo dice frase por frase con Qwen3-TTS, controla cada frase con Whisper (si no
se entiende la repite) y arma un mp3 en <carpeta>/salida/textos/. Las frases que no cambiaron no se rehacen.
"""
import datetime
import re
import shutil
from pathlib import Path

import numpy as np

import paso3_doblar as p3
from comun import (ORIGENES, carpeta_proceso, carpeta_salida, ejecutar_principal, escribir_json, leer_json, log,
                   normalizar_palabras, progreso)

PAUSA_FRASE, PAUSA_PARRAFO, PAUSA_COMA = 0.32, 0.75, 0.18
MAX_PALABRAS = 32

PALABRAS = {  # palabras muy comunes de cada idioma (sin acentos), para adivinar en que esta escrito
    "es": "el la los las de del que y en un una por con para es se no lo como mas pero su sus al le ya muy "
          "tambien porque cuando donde hay esta este esto eso yo mi me nos ellos ella fue son sin sobre hasta "
          "todo todos entre ser tiene hace bien asi".split(),
    "en": "the of and to in is it you that he was for on are with as i his they be at have this from or had "
          "by not but what all were we when your can there an which she do how their if will would about "
          "out them these so her him has my than who been".split(),
    "de": "der die und den von zu das mit sich des auf fur ist im dem nicht ein eine als auch es an werden aus "
          "er hat dass sie nach wird bei einer um am sind noch wie einem uber einen so zum war haben nur oder "
          "aber vor zur bis mehr durch man sein ich du wir ihr".split(),
    "fr": "le la les de des du un une et est en que qui dans ce il elle ne pas pour sur au avec se plus par son "
          "sont mais nous vous ils je tu cette aux leur ou".split(),
    "it": "il lo la gli le di da del della che e un una per non con sono si mi ti ma anche come questo questa "
          "ho hai ha nel alla dei delle sei siamo".split(),
    "pt": "o a os as de do da dos das que e um uma para com nao em no na se por mais mas como eu voce ele ela "
          "nos sao esta foi isso muito tambem".split(),
}
PALABRAS = {cod: set(lista) for cod, lista in PALABRAS.items()}


def detectar_idioma(texto):
    if re.search("[а-яА-ЯёЁ]", texto):
        return "ru"
    palabras = normalizar_palabras(texto)
    if not palabras:
        return "es"
    puntos = {cod: sum(1 for p in palabras if p in conjunto) for cod, conjunto in PALABRAS.items()}
    # letras propias de cada idioma (antes de sacar acentos)
    t = texto.lower()
    puntos["es"] += 3 * len(re.findall("[ñ¿¡]", t))
    puntos["de"] += 3 * len(re.findall("[äöüß]", t))
    puntos["pt"] += 3 * len(re.findall("[ãõç]|ção", t))
    puntos["fr"] += 2 * len(re.findall("[èêëàâîïûœ]", t))
    mejor = max(puntos, key=puntos.get)
    return mejor if puntos[mejor] > 0 else "es"


def frases(texto):
    """[(frase, pausa_despues)]: por parrafos y oraciones; las oraciones largas se cortan en comas."""
    out = []
    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto.replace("\r", "")) if p.strip()]
    for np_, par in enumerate(parrafos):
        par = re.sub(r"\s+", " ", par)
        oraciones = [o.strip() for o in re.split(r"(?<=[.!?…;:])\s+", par) if o.strip()]
        trozos = []
        for o in oraciones:
            if len(o.split()) <= MAX_PALABRAS:
                trozos.append((o, PAUSA_FRASE))
                continue
            actual = ""
            for parte in re.split(r"(?<=,)\s+", o):
                if actual and len((actual + " " + parte).split()) > MAX_PALABRAS:
                    trozos.append((actual, PAUSA_COMA))
                    actual = parte
                else:
                    actual = (actual + " " + parte).strip()
            while len(actual.split()) > MAX_PALABRAS:  # sin comas: a la fuerza
                w = actual.split()
                trozos.append((" ".join(w[:MAX_PALABRAS]), PAUSA_COMA))
                actual = " ".join(w[MAX_PALABRAS:])
            if actual:
                trozos.append((actual, PAUSA_FRASE))
        # oraciones muy cortas se juntan con la siguiente (suenan mejor y salen mas parejas)
        juntas = []
        for t, pausa in trozos:
            if juntas and len(juntas[-1][0].split()) < 4 and len((juntas[-1][0] + " " + t).split()) <= MAX_PALABRAS:
                juntas[-1] = (juntas[-1][0] + " " + t, pausa)
            else:
                juntas.append((t, pausa))
        if juntas and np_ < len(parrafos) - 1:
            juntas[-1] = (juntas[-1][0], PAUSA_PARRAFO)
        out += juntas
    return out


def nombre_archivo(texto):
    primeras = " ".join(re.sub(r"[^\w\s-]", "", texto, flags=re.UNICODE).split()[:6])[:48].strip() or "texto"
    return f"{datetime.datetime.now():%Y-%m-%d %H%M} - {primeras}"


def main(cfg):
    import soundfile as sf

    ruta = Path(cfg["texto_archivo"])
    texto = ruta.read_text(encoding="utf-8").strip()
    if not texto:
        raise SystemExit("El texto esta vacio")
    cod = cfg.get("texto_idioma") or "auto"
    if cod == "auto":
        cod = detectar_idioma(texto)
        log(f"Idioma del texto: {ORIGENES[cod][1]} (detectado)")
    else:
        log(f"Idioma del texto: {ORIGENES[cod][1]}")
    cfg.update(idioma_codigo=cod, idioma_destino=ORIGENES[cod][0])
    p3.ISO3 = ORIGENES[cod][2]

    m = Path(cfg.get("voz_muestra") or "")
    if not ((m / "ref.wav").exists() and (m / "ref.txt").exists()):
        raise SystemExit("No hay muestra de voz preparada en esta carpeta")
    ref_wav, ref_txt = m / "ref.wav", (m / "ref.txt").read_text(encoding="utf-8").strip()

    lista = frases(texto)
    palabras = sum(len(f.split()) for f, _ in lista)
    log(f"{len(lista)} frase(s), {palabras} palabras (~{palabras / 2.6 / 60:.1f} min de audio)")
    w = carpeta_proceso(cfg) / "textos" / cod
    w.mkdir(parents=True, exist_ok=True)
    # cada frase tiene siempre el mismo numero: si se edita el texto, solo se rehacen las frases que cambian
    ids_p = w / "ids.json"
    ids = leer_json(ids_p) if ids_p.exists() else {}
    for f, _ in lista:
        if f not in ids:
            ids[f] = max(ids.values(), default=0) + 1
    escribir_json(ids_p, ids)
    bloques = [{"id": ids[f], "n": k + 1, "en": f} for k, (f, _) in enumerate(lista)]
    motor = p3.Motor(cfg)

    def avance(i, n):
        progreso("texto", "decir", 100.0 * i / max(1, n), f"frase {i + 1} de {n}")

    clips, informe = p3.generar_bloques(motor, w, bloques, cfg, ref_wav, ref_txt, avance)
    progreso("texto", "decir", 96, "armando el audio")

    partes = [np.zeros(int(0.25 * p3.SR), np.float32)]
    for y, (_, pausa) in zip(clips, lista):
        y = np.asarray(y, np.float32)
        f = min(int(0.008 * p3.SR), len(y) // 4)
        if f > 0:
            y[:f] *= np.linspace(0, 1, f)
            y[-f:] *= np.linspace(1, 0, f)
        partes += [y, np.zeros(int(pausa * p3.SR), np.float32)]
    partes.append(np.zeros(int(0.3 * p3.SR), np.float32))
    y = np.concatenate(partes)
    crudo, norm = w / "_texto.wav", w / "_texto_norm.wav"
    sf.write(str(crudo), y, p3.SR)
    try:
        p3.loudnorm(crudo, norm, float(cfg.get("texto_lufs", -16.0)))
    except Exception:
        shutil.copyfile(crudo, norm)
    sal = carpeta_salida(cfg) / "textos"
    sal.mkdir(parents=True, exist_ok=True)
    nombre = nombre_archivo(texto)
    final = p3.exportar_audio(norm, sal / f"{nombre}.mp3", titulo=" ".join(texto.split()[:10]))
    (sal / f"{nombre}.txt").write_text(texto + "\n", encoding="utf-8")
    flojas = [(f, r) for (f, _), r in zip(lista, informe) if (r.get("wer") or 0) > float(cfg.get("wer_max", 0.2))]
    escribir_json(carpeta_proceso(cfg) / "textos" / "ultimo.json",
                  {"archivo": str(final), "idioma": cod, "frases": len(lista),
                   "flojas": [{"frase": f, "oido": r.get("oido")} for f, r in flojas]})
    dur = len(y) / p3.SR
    log(f"Listo: {final} ({dur / 60:.1f} min)")
    if flojas:
        log(f"  {len(flojas)} frase(s) no salieron del todo claras: escuchalas (o reescribilas y generá de nuevo; "
            "solo se rehacen las que cambian)")
    progreso("texto", "decir", 100, "listo")


if __name__ == "__main__":
    ejecutar_principal("TEXTO PARA TU VOZ", main)
