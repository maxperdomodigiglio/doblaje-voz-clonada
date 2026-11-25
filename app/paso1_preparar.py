"""PASO 1 - Preparar videos nuevos.

Para cada video de la carpeta que todavia no fue preparado:
  1. extrae el audio
  2. separa voz y fondo (musica/ambiente) con Demucs
  3. transcribe la voz con Whisper (palabra por palabra)
  4. arma bloques de doblaje y elige un fragmento limpio de referencia para clonar la voz
Deja todo en <carpeta>/_proceso/<capNN>/  (es.json, es_auto.srt, ref.wav, ref.txt, voz.flac, fondo.flac)
"""
import sys
import zlib

import numpy as np

from comun import (APP_DIR, progreso, carpeta_proceso, duracion_media, ejecutar_principal, escribir_json, escribir_srt,
                   ffmpeg, leer_json, listar_videos, log, log_error, slug_de, texto_a_cues)

FORZAR = "--forzar" in sys.argv


# ---------------------------------------------------------------- separacion
def separar(wav_in, voz_out, fondo_out, cfg):
    import soundfile as sf
    import torch

    _orig_load = torch.load

    def _load(*a, **k):  # demucs guarda modelos con pickle completo (pesos oficiales de Meta)
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    torch.load = _load
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nombre = cfg.get("demucs_modelo", "htdemucs_ft")
    log(f"  Demucs ({nombre}) en {dev}...")
    model = get_model(nombre)
    model.eval()

    x, sr = sf.read(str(wav_in), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.T.copy())
    if sr != model.samplerate:
        import julius
        x = julius.resample_frac(x, sr, model.samplerate)
        sr = model.samplerate
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    x = x[:2]
    ref = x.mean(0)
    m, s = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((x - m) / s)[None], device=dev, shifts=1, split=True, overlap=0.25,
                          progress=True)[0]
    out = out.cpu() * s + m
    vi = model.sources.index("vocals")
    voz = out[vi]
    fondo = x - voz  # residuo: todo lo que no es voz (musica, ambiente)
    sf.write(str(voz_out), voz.mean(0).numpy(), sr, format="FLAC", subtype="PCM_16")
    sf.write(str(fondo_out), fondo.numpy().T, sr, format="FLAC", subtype="PCM_16")
    del model, out
    if dev == "cuda":
        torch.cuda.empty_cache()
    torch.load = _orig_load


# ---------------------------------------------------------------- transcripcion
ALUCINACIONES = ("amara.org", "subtítulos realizados", "subtitulos realizados", "suscríbete", "suscribete",
                 "gracias por ver", "¡gracias!", "subtítulos por la comunidad")


def _norm(t):
    import re
    return " ".join(re.findall(r"\w+", (t or "").lower()))


def _segmentos_de(r, offset=0.0, filtrar=True, eco=None):
    out = []
    for sg in r["segments"]:
        txt = sg["text"].strip()
        if filtrar and sg.get("no_speech_prob", 0) > 0.7 and sg.get("avg_logprob", 0) < -1.0:
            continue  # probable alucinacion sobre silencio/musica
        if any(a in txt.lower() for a in ALUCINACIONES):
            continue
        if eco and _norm(txt) and _norm(txt) in _norm(eco):
            continue  # Whisper repitiendo el texto de contexto en vez de transcribir
        palabras = [{"w": w["word"].strip(), "s": round(float(w["start"]) + offset, 3),
                     "e": round(float(w["end"]) + offset, 3), "p": round(float(w.get("probability", 1.0)), 3)}
                    for w in sg.get("words", []) if w["word"].strip()]
        if not palabras:
            continue
        d = palabras[-1]["e"] - palabras[0]["s"]
        if len(palabras) >= 3 and len(palabras) / max(d, 0.05) > 7:
            continue  # palabras amontonadas en un instante: alucinacion
        out.append({"start": palabras[0]["s"], "end": palabras[-1]["e"], "text": txt,
                    "avg_logprob": round(sg.get("avg_logprob", 0), 3),
                    "no_speech_prob": round(sg.get("no_speech_prob", 0), 3), "words": palabras})
    return out


def _sospechoso(sg, y=None, sr=16000):
    """Segmento estirado: pocas palabras para el tiempo con voz que abarca (Whisper se 'comio' parte del habla)."""
    d = sg["end"] - sg["start"]
    if d <= 2.5:
        return False
    activo = d
    if y is not None:
        hop = int(0.25 * sr)
        z = y[int(sg["start"] * sr):int(sg["end"] * sr)]
        n = len(z) // hop
        if n:
            e = 20 * np.log10(np.sqrt(np.mean(z[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
            activo = max(0.25, float((e > -38).sum()) * 0.25)
    return len(sg["words"]) / activo < 1.2


def _huecos(y, segmentos, sr=16000, umbral_db=-38.0):
    """Zonas con voz (en la pista de voz separada) que no quedaron transcriptas."""
    hop = int(0.25 * sr)
    n = len(y) // hop
    e = 20 * np.log10(np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    cubierto = np.zeros(n, bool)
    for sg in segmentos:
        if _sospechoso(sg, y, sr):
            continue
        for w in sg["words"]:
            a, b = int((w["s"] - 0.3) / 0.25), int((w["e"] + 0.3) / 0.25) + 1
            cubierto[max(0, a):min(n, b)] = True
    act = (e > umbral_db) & ~cubierto
    runs, i = [], 0
    while i < n:
        if act[i]:
            j = i
            while j < n and (act[j] or (j + 2 < n and act[j + 1:j + 3].any())):
                j += 1
            if (j - i) * 0.25 >= 0.75:
                runs.append((i * 0.25, j * 0.25))
            i = j
        else:
            i += 1
    return runs


def _ventanas(runs, segmentos, dur, y=None):
    vents = []
    for a, b in runs:
        prev = max([sg["end"] for sg in segmentos if sg["end"] <= a and not _sospechoso(sg, y)] + [0.0])
        nxt = min([sg["start"] for sg in segmentos if sg["start"] >= b and not _sospechoso(sg, y)] + [dur])
        # arrancar cerca de donde empieza la voz: si la ventana incluye muchos segundos de musica/placa
        # de titulo, Whisper tiende a inventar o a saltearse el habla
        vents.append([max(prev, a - 0.6), min(nxt, b + 0.6)])
    vents.sort()
    fus = []
    for v in vents:
        if fus and v[0] <= fus[-1][1] + 1.5:
            fus[-1][1] = max(fus[-1][1], v[1])
        else:
            fus.append(v)
    return fus


IDIOMA_DETECTADO = None


def _detectar_idioma(model, y):
    """Idioma del habla: vota en 3 tramos de 30 s (a 1/4, 1/2 y 3/4), asi una cancion al principio no confunde."""
    import whisper
    n = 30 * 16000
    total = {}
    for frac in ((0.5,) if len(y) <= n else (0.25, 0.5, 0.75)):
        i0 = max(0, min(len(y) - n, int(len(y) * frac) - n // 2))
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(y[i0:i0 + n]), n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        for k, v in probs.items():
            total[k] = total.get(k, 0.0) + v
    return max(total, key=total.get)


def transcribir(voz_path, cfg):
    import librosa
    import torch
    import whisper

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nombre = cfg.get("whisper_modelo", "large-v3")
    log(f"  Whisper ({nombre}) en {dev}...")
    model = whisper.load_model(nombre, device=dev, download_root=str(APP_DIR / "modelos" / "whisper"))
    y, _ = librosa.load(str(voz_path), sr=16000, mono=True)
    y = y.astype(np.float32)
    dur = len(y) / 16000
    global IDIOMA_DETECTADO
    idioma = cfg.get("idioma_origen", "es")
    if not idioma or idioma == "auto":
        try:
            idioma = _detectar_idioma(model, y)
        except Exception:
            idioma = None  # que lo detecte Whisper al transcribir
    opts = dict(language=idioma, task="transcribe", word_timestamps=True,
                fp16=(dev == "cuda"))
    prompt = cfg.get("whisper_prompt") or None
    r = model.transcribe(y, condition_on_previous_text=False, initial_prompt=prompt, verbose=False, **opts)
    if not idioma:
        idioma = r.get("language") or "es"
        opts["language"] = idioma
    if idioma != cfg.get("idioma_origen"):
        log(f"  Idioma del audio: {idioma}")
    IDIOMA_DETECTADO = idioma
    segmentos = _segmentos_de(r, eco=prompt)

    # segunda pasada: volver a transcribir las zonas con voz que Whisper salteo
    for vuelta in range(2):
        runs = _huecos(y, segmentos)
        if not runs:
            break
        vents = _ventanas(runs, segmentos, dur, y)
        log(f"  Rellenando {len(vents)} hueco(s) sin transcribir: " +
            ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in vents))
        for a, b in vents:
            a0, b0 = max(0.0, a - 0.15), min(dur, b + 0.15)
            previo = " ".join(sg["text"] for sg in segmentos if sg["end"] <= a)[-200:] or None
            extra = {} if vuelta == 0 else {"beam_size": 5, "patience": 1.0}  # 2da vuelta: otra decodificacion
            rr = model.transcribe(y[int(a0 * 16000):int(b0 * 16000)], condition_on_previous_text=True,
                                  initial_prompt=previo, verbose=None, **extra, **opts)
            nuevos = [sg for sg in _segmentos_de(rr, offset=a0, filtrar=False, eco=previo)
                      if sg["end"] > a - 0.05 and sg["start"] < b + 0.05]
            if not nuevos:
                continue
            segmentos = [sg for sg in segmentos if not (a - 0.05 <= (sg["start"] + sg["end"]) / 2 <= b + 0.05)]
            segmentos = sorted(segmentos + nuevos, key=lambda sg: sg["start"])

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    # descartar "frases" sobre silencio de la pista de voz (alucinaciones sobre musica/placas)
    def _energia(a, b):
        z = y[int(a * 16000):int(b * 16000)]
        return 20 * np.log10(np.sqrt(np.mean(z ** 2)) + 1e-9) if len(z) else -120.0

    antes = len(segmentos)
    segmentos = [sg for sg in segmentos if _energia(sg["start"], sg["end"]) > -42]
    if len(segmentos) < antes:
        log(f"  Descarto {antes - len(segmentos)} segmento(s) sin voz real (alucinaciones)")
    for i, sg in enumerate(segmentos, 1):
        sg["id"] = i
    return segmentos


# ---------------------------------------------------------------- bloques de doblaje
FIN = (".", "?", "!", "…", ".\"", "?\"", "!\"")
PAUSA_SUAVE = (",", ";", ":")


def armar_bloques(segmentos, max_dur=14.0):
    palabras = [w for sg in segmentos for w in sg["words"]]
    bloques, cur = [], []

    def cerrar():
        if cur:
            bloques.append({"id": len(bloques) + 1, "start": cur[0]["s"], "end": cur[-1]["e"],
                            "es": " ".join(w["w"] for w in cur)})

    for w in palabras:
        if cur:
            gap = w["s"] - cur[-1]["e"]
            dur = cur[-1]["e"] - cur[0]["s"]
            ult = cur[-1]["w"]
            cortar = (gap >= 0.9
                      or (gap >= 0.3 and ult.endswith(FIN) and dur >= 1.5)
                      or (dur >= max_dur and (ult.endswith(FIN + PAUSA_SUAVE) or gap >= 0.2))
                      or dur >= max_dur * 1.5)
            if cortar:
                cerrar()
                cur = []
        cur.append(w)
    cerrar()
    return bloques


def elegir_referencia(bloques, segmentos, voz_path, d, cfg):
    """Busca 8-14 s de habla continua, con buena confianza de Whisper, para clonar la voz."""
    import librosa
    import soundfile as sf

    palabras = [w for sg in segmentos for w in sg["words"]]
    mejor, mejor_score = None, -1e9
    for i in range(len(palabras)):
        for j in range(i + 1, len(palabras)):
            dur = palabras[j]["e"] - palabras[i]["s"]
            if dur > 14.5:
                break
            if dur < 7.5:
                continue
            tramo = palabras[i:j + 1]
            gaps = [b["s"] - a["e"] for a, b in zip(tramo, tramo[1:])]
            if max(gaps, default=0) > 0.7:
                break
            # que empiece despues de una pausa y termine en fin de frase si se puede
            antes = palabras[i]["s"] - palabras[i - 1]["e"] if i > 0 else 1.0
            despues = palabras[j + 1]["s"] - palabras[j]["e"] if j + 1 < len(palabras) else 1.0
            conf = float(np.mean([w["p"] for w in tramo]))
            score = conf * 10 - abs(dur - 11) * 0.3 + min(antes, 0.6) + min(despues, 0.6)
            if tramo[-1]["w"].endswith(FIN):
                score += 1.0
            if despues < 0.35:
                score -= 5.0  # sin pausa despues: el clon tiende a "arrastrar" la ultima palabra
            if score > mejor_score:
                mejor, mejor_score = (i, j), score
    cola = 0.18
    if mejor is None:  # fallback: el bloque mas largo
        b = max(bloques, key=lambda b: b["end"] - b["start"])
        t0, t1, texto = b["start"], min(b["end"], b["start"] + 15), b["es"]
    else:
        i, j = mejor
        t0, t1 = palabras[i]["s"], palabras[j]["e"]
        texto = " ".join(w["w"] for w in palabras[i:j + 1])
        despues = palabras[j + 1]["s"] - palabras[j]["e"] if j + 1 < len(palabras) else 1.0
        cola = max(0.15, min(0.45, despues - 0.1))  # que la ultima palabra entre completa

    y, sr = librosa.load(str(voz_path), sr=24000, mono=True, offset=max(0, t0 - 0.12), duration=(t1 - t0) + 0.12 + cola)
    fade = int(0.02 * sr)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    pico = np.max(np.abs(y)) or 1.0
    y = y / pico * 0.89
    sf.write(str(d / "ref.wav"), y, sr)
    (d / "ref.txt").write_text(texto, encoding="utf-8")
    return {"start": round(t0, 3), "end": round(t1, 3), "texto": texto}


# ---------------------------------------------------------------- principal

def rehacer_ventanas(d, cfg):
    """Re-transcribe tramos puntuales pedidos en _proceso/capNN/rehacer.json: [[inicio, fin], ...] (segundos)."""
    import librosa
    import torch
    import whisper

    pedido = d / "rehacer.json"
    es = leer_json(d / "es.json")
    vents = leer_json(pedido)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[{d.name}] Re-transcribiendo tramos pedidos: " + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in vents))
    model = whisper.load_model(cfg.get("whisper_modelo", "large-v3"), device=dev,
                               download_root=str(APP_DIR / "modelos" / "whisper"))
    y, _ = librosa.load(str(d / "voz.flac"), sr=16000, mono=True)
    y = y.astype(np.float32)
    segmentos = es["segmentos"]
    for a, b in vents:
        previo = " ".join(sg["text"] for sg in segmentos if sg["end"] <= a)[-200:] or None
        rr = model.transcribe(y[int(a * 16000):int(b * 16000)], language=es.get("idioma_origen") or "es",
                              task="transcribe", word_timestamps=True, fp16=(dev == "cuda"),
                              condition_on_previous_text=True, initial_prompt=previo, beam_size=5,
                              patience=1.0, verbose=None)
        nuevos = _segmentos_de(rr, offset=a, filtrar=False, eco=previo)
        log(f"  {a:.1f}-{b:.1f}s -> " + " | ".join(sg["text"] for sg in nuevos))
        segmentos = [sg for sg in segmentos if not (a <= (sg["start"] + sg["end"]) / 2 <= b)]
        segmentos = sorted(segmentos + nuevos, key=lambda sg: sg["start"])
    del model
    for i, sg in enumerate(segmentos, 1):
        sg["id"] = i
    es["segmentos"] = segmentos
    es["bloques"] = armar_bloques(segmentos, cfg.get("bloque_max_seg", 14.0))
    escribir_json(d / "es.json", es)
    cues = []
    for sg in segmentos:
        cues += texto_a_cues(sg["text"], sg["start"], sg["end"])
    escribir_srt(d / "es_auto.srt", cues)
    pedido.replace(d / "rehacer_hecho.json")
    log(f"[{d.name}] LISTO re-transcripcion")


VERSION = 3  # sube cuando mejora la transcripcion: los capitulos viejos se re-transcriben solos


def preparar(video, cfg):
    proc = carpeta_proceso(cfg)
    slug = slug_de(video.name)
    d = proc / slug
    if (d / "rehacer.json").exists() and (d / "es.json").exists():
        progreso(slug, 1, 30, "re-escuchando frases dudosas")
        rehacer_ventanas(d, cfg)
        progreso(slug, 1, 100, "listo")
        return
    if (d / "es.json").exists() and not FORZAR:
        previo = leer_json(d / "es.json")
        if previo.get("video") != video.name:
            slug = slug + "_" + str(zlib.crc32(video.name.encode("utf-8")) % 10000)
            d = proc / slug
            previo = leer_json(d / "es.json") if (d / "es.json").exists() else {}
        if previo.get("video") == video.name and previo.get("version", 1) >= VERSION:
            log(f"[{slug}] ya preparado, salteo ({video.name})")
            progreso(slug, 1, 100, "ya estaba preparado")
            return
    d.mkdir(parents=True, exist_ok=True)
    log(f"[{slug}] PREPARANDO: {video.name}")
    progreso(slug, 1, 2, "extrayendo el audio")

    dur = duracion_media(video)
    wav = d / "audio_orig.wav"
    if FORZAR or not ((d / "voz.flac").exists() and (d / "fondo.flac").exists()):
        log("  Extrayendo audio...")
        ffmpeg("-i", video, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", wav)
        progreso(slug, 1, 8, "separando la voz de la música")
        separar(wav, d / "voz.flac", d / "fondo.flac", cfg)
    else:
        log("  Voz y fondo ya separados, reutilizo")
    progreso(slug, 1, 40, "transcribiendo lo que se dice")
    segmentos = transcribir(d / "voz.flac", cfg)
    if not segmentos:
        raise RuntimeError("Whisper no encontro habla en el video")
    progreso(slug, 1, 92, "eligiendo la muestra de voz")
    bloques = armar_bloques(segmentos, cfg.get("bloque_max_seg", 14.0))
    ref = elegir_referencia(bloques, segmentos, d / "voz.flac", d, cfg)
    log(f"  {len(segmentos)} segmentos, {len(bloques)} bloques; referencia de voz {ref['start']:.1f}-{ref['end']:.1f}s")

    escribir_json(d / "es.json", {"version": VERSION, "video": video.name, "slug": slug, "duracion": round(dur, 3),
                                  "idioma_origen": IDIOMA_DETECTADO or "es",
                                  "segmentos": segmentos, "bloques": bloques, "ref": ref})
    cues = []
    for sg in segmentos:
        cues += texto_a_cues(sg["text"], sg["start"], sg["end"])
    escribir_srt(d / "es_auto.srt", cues)
    try:
        wav.unlink(missing_ok=True)
    except OSError:
        pass
    log(f"[{slug}] LISTO paso 1 -> {d}")
    progreso(slug, 1, 100, "listo")


def main(cfg):
    videos = listar_videos(cfg)
    if not videos:
        log("No hay videos en la carpeta.")
        return
    errores = 0
    for v in videos:
        try:
            preparar(v, cfg)
        except Exception:
            errores += 1
            log_error(v.name)
    log(f"Paso 1 terminado. Errores: {errores}")
    if errores:
        sys.exit(1)


if __name__ == "__main__":
    ejecutar_principal("PASO 1 (preparar)", main)
