"""PASO 3 - Doblar y mezclar.

Para cada capitulo con traduccion (_proceso/<capNN>/en.json):
  1. genera cada bloque en ingles con la voz clonada (Qwen3-TTS Base, referencia ref.wav)
  2. controla cada bloque con Whisper (si no dice lo que tiene que decir, lo regenera)
  3. lo ubica en su lugar del video (acelera hasta max_aceleracion si no entra)
  4. mezcla con el fondo original (musica/ambiente) bajandolo un poco debajo de la voz
  5. normaliza el volumen (LUFS) y arma el video final con subtitulos (castellano o el idioma original)
Si la entrada es un audio (mp3, wav, m4a...), sale un audio doblado (mp3) y los subtitulos en .srt.
Resultado en <carpeta>/salida/
"""
import hashlib
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from comun import (ORIGENES, es_audio, idioma_origen, progreso, archivo_traduccion, carpeta_idioma, referencia_voz, APP_DIR, carpeta_proceso, carpeta_salida, correr, ejecutar_principal, escribir_json,
                   escribir_srt, ffmpeg, ffmpeg_exe, leer_json, log, log_error, texto_a_cues, wer)

SR = 44100
FORZAR = "--forzar" in sys.argv
SIMULAR = "--simular" in sys.argv  # pruebas sin GPU: reemplaza la voz por un tono
SOLO = [a for a in sys.argv[1:] if not a.startswith("--")]
SUFIJO, ISO3 = "EN", "eng"  # se pisan con la config (sufijo_salida / idioma_iso3)


def idioma_subs(src, cod):
    """Subtitulos del resultado: en castellano siempre que se pueda (el original si es castellano, la traduccion si
    se dobla al castellano); si no, en el idioma original. Devuelve (usar_traduccion, codigo)."""
    if src != "es" and cod == "es":
        return True, "es"
    return False, src


def _iso3(cod):
    return ORIGENES.get(cod, (cod, cod, cod))[2]


def _nombre_idioma(cod):
    return ORIGENES.get(cod, (cod,))[0]


def exportar_audio(mezcla, destino_mp3, titulo=""):
    """Audio final: mp3 192k (o m4a si este ffmpeg no trae mp3). Devuelve la ruta que quedo."""
    meta = ["-metadata", f"title={titulo}"] if titulo else []
    for dest, codec in ((destino_mp3, ["-c:a", "libmp3lame", "-b:a", "192k"]),
                        (destino_mp3.with_suffix(".m4a"), ["-c:a", "aac", "-b:a", "192k"])):
        tmp = dest.with_name(dest.stem + ".tmp" + dest.suffix)
        try:
            ffmpeg("-i", mezcla, "-map", "0:a:0", *codec, *meta, "-metadata:s:a:0", f"language={ISO3}", tmp)
            tmp.replace(dest)
            return dest
        except RuntimeError:
            tmp.unlink(missing_ok=True)
            if dest is not destino_mp3:
                raise
    return destino_mp3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def estimar_duracion(texto):
    return len(texto.split()) / 2.6 + 0.3


# ---------------------------------------------------------------- motor TTS + control
class Motor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tts = None
        self.asr = None

    def _dev(self):
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def cargar_tts(self):
        if self.tts is not None or SIMULAR:
            return
        import torch
        from qwen_tts import Qwen3TTSModel
        dev = self._dev()
        nombre = self.cfg.get("tts_modelo", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        local = self.cfg.get("tts_modelo_local")
        if local and Path(local).is_dir():
            nombre = local  # reutiliza el modelo que ya bajo la app Qwen3-TTS (ahorra 4 GB)
        log(f"  Cargando {nombre} en {dev} (la primera vez descarga ~4 GB)...")
        self.tts = Qwen3TTSModel.from_pretrained(nombre, device_map=dev,
                                                 dtype=torch.bfloat16 if dev != "cpu" else torch.float32)

    def prompt(self, ref_wav, ref_txt):
        if SIMULAR:
            return None
        import soundfile as sf
        xv = self.cfg.get("modo_clon", "icl") == "xvector"
        y, sr = sf.read(str(ref_wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        return self.tts.create_voice_clone_prompt(ref_audio=(y, sr), ref_text=None if xv else ref_txt,
                                                  x_vector_only_mode=xv)

    def prompt_xvector(self, ref_wav):
        if SIMULAR:
            return None
        import soundfile as sf
        y, sr = sf.read(str(ref_wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        return self.tts.create_voice_clone_prompt(ref_audio=(y, sr), ref_text=None, x_vector_only_mode=True)

    def generar(self, texto, prompt, seed):
        set_seed(seed)
        est = estimar_duracion(texto)
        if SIMULAR:
            dur = est * random.uniform(0.85, 1.3)
            t = np.arange(int(dur * 24000)) / 24000
            y = 0.2 * np.sin(2 * np.pi * 140 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
            return y.astype(np.float32), 24000
        wavs, sr = self.tts.generate_voice_clone(
            text=texto, language=self.cfg.get("idioma_destino", "English"), voice_clone_prompt=prompt,
            max_new_tokens=int(min(2048, 12.5 * (est * 2.5 + 6))))
        return np.asarray(wavs[0], dtype=np.float32), sr

    def escuchar(self, y, sr):
        """Transcribe el clip generado para controlar que diga el texto correcto."""
        if SIMULAR or not self.cfg.get("control_whisper", True):
            return None
        import librosa
        import torch
        import whisper
        if self.asr is None:
            nombre = self.cfg.get("whisper_control", "turbo")
            log(f"  Cargando Whisper de control ({nombre})...")
            self.asr = whisper.load_model(nombre, device=self._dev(), download_root=str(APP_DIR / "modelos" / "whisper"))
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
        r = self.asr.transcribe(y16, language=self.cfg.get("idioma_codigo", "en"), fp16=torch.cuda.is_available(),
                                condition_on_previous_text=False, verbose=None)
        return r["text"].strip()


# ---------------------------------------------------------------- audio helpers
def recortar_silencios(y, sr):
    import librosa
    if len(y) == 0:
        return y
    _, (a, b) = librosa.effects.trim(y, top_db=40, frame_length=1024, hop_length=256)
    m = int(0.06 * sr)
    return y[max(0, a - m):min(len(y), b + m)]


def rms_activo(y, umbral_db=-45):
    if len(y) == 0:
        return 0.0
    fr = 1024
    n = len(y) // fr
    if n == 0:
        return float(np.sqrt(np.mean(y ** 2)))
    e = np.sqrt(np.mean(y[:n * fr].reshape(n, fr) ** 2, axis=1))
    act = e[e > 10 ** (umbral_db / 20)]
    return float(np.sqrt(np.mean(act ** 2))) if len(act) else float(np.sqrt(np.mean(e ** 2)))


def acelerar(y, factor):
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(str(a), y, SR, subtype="FLOAT")
        ffmpeg("-i", a, "-filter:a", f"atempo={factor:.4f}", "-c:a", "pcm_f32le", b)
        z, _ = sf.read(str(b), dtype="float32")
    return z


def suavizar(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def ducking(fondo, voz, db):
    """Baja el fondo 'db' decibeles mientras habla la voz doblada (con ataque/relajacion suaves)."""
    if db <= 0:
        return fondo
    hop = 441  # 10 ms
    n = len(voz) // hop
    e = np.sqrt(np.mean(voz[:n * hop].reshape(n, hop) ** 2, axis=1))
    act = (e > 10 ** (-45 / 20)).astype(float)
    act = np.clip(suavizar(act, 31) * 3, 0, 1)  # dilata ~150 ms
    act = suavizar(act, 15)
    g = 1 - (1 - 10 ** (-db / 20)) * act
    gain = np.interp(np.arange(len(fondo)), np.arange(n) * hop + hop / 2, g, left=1.0, right=1.0)
    return fondo * gain[:, None].astype(np.float32)


def loudnorm(entrada, salida, lufs, tp=-1.5):
    import json
    import re
    f1 = f"loudnorm=I={lufs}:TP={tp}:LRA=11:print_format=json"
    r = correr([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(entrada), "-af", f1, "-f", "null", "-"])
    m = re.findall(r"\{[^{}]*\}", r.stderr, re.S)
    j = json.loads(m[-1])
    f2 = (f"loudnorm=I={lufs}:TP={tp}:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}"
          f":measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}"
          f":linear=true")
    ffmpeg("-i", entrada, "-af", f2, "-ar", "48000", "-c:a", "pcm_s16le", salida)
    return float(j["input_i"])


ZOCALO = (200, 515, 1072, 651)  # zona del cartel "Doctor en Psicologia - Gregorio Dunayevich" (en 1280x720)


def detectar_zocalos(video, d):
    """Detecta cuando aparece el cartel con el nombre abajo, para subir los subtitulos en ese momento."""
    import json
    import subprocess
    cache = d / "zocalos2.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    tpl_p = APP_DIR / "zocalos.npy"  # carteles: nombre de Gregorio y "Comentarios y preguntas"
    if not tpl_p.exists():
        tpl_p = APP_DIR / "zocalo.npy"
        if not tpl_p.exists():
            return []
    tpls = np.load(str(tpl_p))
    if tpls.ndim == 2:
        tpls = tpls[None]
    x0, y0, x1, y1 = ZOCALO
    sh, sw = tpls.shape[1:]
    fps = 2
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf",
           f"fps={fps},scale=1280:720,crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={sw}:{sh},format=gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (sw * sh)
    frames = np.frombuffer(raw[:n * sw * sh], np.uint8).reshape(n, sh, sw).astype(np.float32)
    ts = [t - t.mean() for t in tpls]
    rangos = []
    for i, f in enumerate(frames):
        g = f - f.mean()
        c = max(float((g * t).sum() / (np.sqrt((g * g).sum() * (t * t).sum()) + 1e-6)) for t in ts)
        if c > 0.55:
            a, b = i / fps - 0.6, i / fps + 0.6
            if rangos and a <= rangos[-1][1]:
                rangos[-1][1] = b
            else:
                rangos.append([a, b])
    cache.write_text(json.dumps(rangos), encoding="utf-8")
    if rangos:
        log("  Cartel con nombre detectado en: " + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in rangos))
    return rangos


def srt_a_ass(srt_path, ass_path, ancho=1280, alto=720, arriba=()):
    import re
    txt = Path(srt_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    bloques = re.split(r"\n\s*\n", txt.strip())
    fs = round(alto * 0.055)
    lineas = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {ancho}", f"PlayResY: {alto}", "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Default,Arial,{fs},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,"
        f"{max(2, round(fs * 0.07))},1,2,60,60,{round(alto * 0.05)},1",
        f"Style: Arriba,Arial,{fs},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,"
        f"{max(2, round(fs * 0.07))},1,8,60,60,{round(alto * 0.05)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def t_ass(t):
        h, m, rest = t.split(":")
        sec, ms = rest.split(",")
        return f"{int(h)}:{int(m):02d}:{int(sec):02d}.{int(ms) // 10:02d}"

    def seg(t):
        h, m, rest = t.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    for b in bloques:
        partes = b.strip().split("\n")
        if len(partes) < 3 or "-->" not in partes[1]:
            continue
        a, z = [x.strip() for x in partes[1].split("-->")]
        texto = "\\N".join(p.replace("{", "(").replace("}", ")") for p in partes[2:])
        estilo = "Arriba" if any(seg(a) < r1 and seg(z) > r0 for r0, r1 in arriba) else "Default"
        lineas.append(f"Dialogue: 0,{t_ass(a)},{t_ass(z)},{estilo},,0,0,0,,{texto}")
    Path(ass_path).write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _ffmpegs_con_libass():
    """ffmpeg a probar para grabar subtitulos: primero el de imageio (trae libass seguro), despues el del sistema."""
    exes = []
    try:
        import imageio_ffmpeg
        exes.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    sistema = ffmpeg_exe()
    if sistema not in exes:
        exes.append(sistema)
    return exes


def quemar_subtitulos(video, mezcla, srt_es, destino, d, arriba=None):
    log("  Grabando subtitulos en la imagen...")
    if arriba is None:
        arriba = detectar_zocalos(video, d)
    srt_a_ass(srt_es, d / "_quemar.ass", arriba=arriba)
    tmp = destino.with_suffix(".tmp.mp4")
    errores = []
    for exe in _ffmpegs_con_libass():
        for filtro in ("ass=filename=_quemar.ass", "subtitles=filename=_quemar.ass", "ass=_quemar.ass"):
            for codec in CODECS_VIDEO:
                try:
                    correr([exe, "-hide_banner", "-nostdin", "-y", "-i", video, "-i", mezcla,
                            "-map", "0:v:0", "-map", "1:a:0", "-vf", filtro, *codec,
                            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", tmp], cwd=str(d))
                    tmp.replace(destino)
                    return
                except RuntimeError as e:
                    errores.append(f"{Path(exe).name} [{filtro}] {codec[1]}: " + str(e).strip().splitlines()[-1])
    raise RuntimeError("No pude grabar los subtitulos:\n" + "\n".join(errores))


# ---------------------------------------------------------------- generacion por bloque
def palabra_extra_inicial(texto, oido, ref_txt):
    """True si lo generado arranca con una palabra que no esta en el texto (p.ej. el final de la referencia)."""
    from comun import normalizar_palabras
    oi, tx, rf = normalizar_palabras(oido or ""), normalizar_palabras(texto), normalizar_palabras(ref_txt)
    if not oi or not tx or oi[0] == tx[0]:
        return False
    return (rf and oi[0] == rf[-1]) or (len(oi) > 1 and oi[1] == tx[0])


def generar_bloques(motor, w, bloques, cfg, ref_wav, ref_txt, avance=None):
    import librosa
    import soundfile as sf

    tts_dir = w / "tts"
    tts_dir.mkdir(exist_ok=True)
    ref_hash = hashlib.md5(ref_wav.read_bytes() + ref_txt.encode()).hexdigest()[:10]
    clips, informe = [], []
    prompt = None
    prompt_xv = None
    intentos_max = int(cfg.get("intentos_max", 3))
    for i, b in enumerate(bloques):
        if avance:
            avance(i, len(bloques))
        texto = (b.get("tts") or b["en"]).strip()
        h = hashlib.md5((texto + ref_hash + cfg.get("modo_clon", "icl") + str(SIMULAR)).encode()).hexdigest()[:12]
        wav_p, meta_p = tts_dir / f"b{b['id']:03d}.wav", tts_dir / f"b{b['id']:03d}.json"
        if wav_p.exists() and meta_p.exists() and leer_json(meta_p).get("hash") == h:
            y, sr = sf.read(str(wav_p), dtype="float32")
            meta = leer_json(meta_p)
        else:
            motor.cargar_tts()
            if prompt is None and not SIMULAR:
                prompt = motor.prompt(ref_wav, ref_txt)
            est = estimar_duracion(texto)
            mejor = None
            fugas = 0
            for intento in range(intentos_max + 1):
                seed = int(cfg.get("semilla", 1234)) + b["id"] * 100 + intento
                usar_xv = fugas >= 2 or intento == intentos_max  # respaldo sin "continuar" la referencia
                if usar_xv and prompt_xv is None and not SIMULAR:
                    prompt_xv = motor.prompt_xvector(ref_wav)
                y, sr = motor.generar(texto, prompt_xv if usar_xv else prompt, seed)
                y = recortar_silencios(y, sr)
                dur = len(y) / sr
                oido = motor.escuchar(y, sr)
                # comparar contra el texto hablado y contra el subtitulo (Whisper escribe "27" en vez de "twenty-seven")
                err = min(wer(texto, oido), wer(b["en"], oido)) if oido is not None else 0.0
                fuga = (palabra_extra_inicial(texto, oido, ref_txt) and palabra_extra_inicial(b["en"], oido, ref_txt))
                fugas += int(fuga)
                malo_dur = dur < est * 0.45 or dur > est * 2.2 + 1.0
                score = err * 3 + (2.0 if malo_dur else 0.0) + (3.0 if fuga else 0.0) + (0.3 if usar_xv else 0.0)
                cand = {"y": y, "sr": sr, "seed": seed, "dur": round(dur, 2), "wer": round(err, 3),
                        "oido": oido, "score": score, "intento": intento + 1, "fuga": fuga, "xvector": usar_xv}
                if mejor is None or score < mejor["score"]:
                    mejor = cand
                if err <= float(cfg.get("wer_max", 0.2)) and not malo_dur and not fuga:
                    break
                if intento == intentos_max - 1 and not fugas and mejor["score"] < 2:
                    break  # sin fugas: no hace falta el intento extra
                motivo = "palabra extra al inicio" if fuga else f"wer {err:.2f}, {dur:.1f}s"
                log(f"    bloque {b['id']}: intento {intento + 1} flojo ({motivo}) -> reintento")
            y, sr = mejor["y"], mejor["sr"]
            meta = {k: v for k, v in mejor.items() if k not in ("y", "sr")}
            meta.update(hash=h, texto=texto)
            sf.write(str(wav_p), y, sr)
            escribir_json(meta_p, meta)
            log(f"  bloque {b['id']:3d}/{len(bloques)}  {meta['dur']:5.1f}s  wer {meta['wer']:.2f}  {texto[:60]}")
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
        clips.append(y)
        informe.append({"id": b["id"], **{k: meta.get(k) for k in ("dur", "wer", "intento", "seed", "oido", "fuga", "xvector")}})
    return clips, informe


# ---------------------------------------------------------------- ensamblado
def _actividad(y, sr=SR, paso=0.01, umbral_db=-40.0, relativo=None):
    hop = max(1, int(paso * sr))
    k = len(y) // hop
    if k == 0:
        return np.zeros(0, bool)
    e = 20 * np.log10(np.sqrt(np.mean(y[:k * hop].reshape(k, hop) ** 2, axis=1)) + 1e-9)
    th = umbral_db if relativo is None else max(umbral_db, float(np.max(e)) - relativo)
    return e > th


def frases_del_clip(clip, min_sil=0.14, min_frase=0.25):
    """Corta el clip TTS en frases por sus pausas internas. Devuelve [(ini_muestra, fin_muestra)]."""
    act = _actividad(clip, relativo=38.0)
    n = len(act)
    if n == 0:
        return [(0, len(clip))]
    cortes, i = [], 0
    while i < n:
        if not act[i]:
            j = i
            while j < n and not act[j]:
                j += 1
            if (j - i) * 0.01 >= min_sil and i > 0 and j < n:
                cortes.append(((i + j) // 2) * int(0.01 * SR))
            i = j
        else:
            i += 1
    bordes = [0] + cortes + [len(clip)]
    frases = [(a, b) for a, b in zip(bordes, bordes[1:]) if b > a]
    # unir frases muy cortas con la anterior
    out = []
    for a, b in frases:
        if out and (b - a) / SR < min_frase:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def distribuir(clip, voz_tramo, t0, cfg, limite=None):
    """Reparte las frases del clip sobre los momentos en que Gregorio habla en el original
    (mapeo proporcional de tiempo hablado), para no dejar silencios largos mientras mueve la boca.
    Devuelve [(tiempo_inicio, audio)]."""
    frases = frases_del_clip(clip)
    act = _actividad(voz_tramo, umbral_db=-40.0)
    if len(frases) < 2 or act.sum() < 10:
        return [(t0, clip)]
    acum = np.cumsum(act.astype(float))
    total_orig = acum[-1]
    dur_frases = [(b - a) / SR for a, b in frases]
    total_tts = sum(dur_frases)
    piezas, cursor, previo = [], t0, 0.0
    gap_min = 0.12
    for (a, b), dfr in zip(frases, dur_frases):
        x = previo / total_tts
        idx = int(np.searchsorted(acum, x * total_orig))
        objetivo = t0 + idx * 0.01
        inicio = max(objetivo, cursor + (gap_min if piezas else 0.0))
        trozo = clip[a:b].copy()
        f = min(len(trozo) // 2, int(0.005 * SR))
        if f > 0:
            trozo[:f] *= np.linspace(0, 1, f)
            trozo[-f:] *= np.linspace(1, 0, f)
        piezas.append((inicio, trozo))
        cursor = inicio + dfr
        previo += dfr
    if limite is not None and cursor > limite:
        # no pisar la frase siguiente: correr hacia atras las piezas (sin arrancar antes de t0)
        tope = limite
        nuevas = []
        for inicio, trozo in reversed(piezas):
            dur = len(trozo) / SR
            inicio = min(inicio, tope - dur)
            nuevas.append((inicio, trozo))
            tope = inicio - gap_min
        nuevas.reverse()
        if nuevas[0][0] < t0:  # no entra repartido: todo seguido desde t0
            return [(t0, clip)]
        piezas = nuevas
    return piezas


def ensamblar(bloques, clips, voz_orig, dur_total, cfg):
    n_total = int(round(dur_total * SR))
    pista = np.zeros(n_total + SR * 10, np.float32)
    max_acel = float(cfg.get("max_aceleracion", 1.2))
    min_tempo = float(cfg.get("min_tempo", 0.9))  # cuanto se puede "frenar" una frase que queda corta
    repartir = cfg.get("repartir_frases", True)
    # nivel: igualar el nivel de la voz original de cada bloque (respeta los enfasis), acotado +-6 dB
    niveles = []
    for b in bloques:
        a, z = int(b["start"] * SR), int(b["end"] * SR)
        niveles.append(rms_activo(voz_orig[a:z]))
    med = float(np.median([n for n in niveles if n > 0]) or 0.1)
    cursor, colocados = 0.0, []
    for i, (b, clip) in enumerate(zip(bloques, clips)):
        objetivo = float(np.clip(niveles[i] or med, med * 0.5, med * 2.0))
        r = rms_activo(clip)
        if r > 0:
            clip = clip * (objetivo / r)
        sig = bloques[i + 1]["start"] if i + 1 < len(bloques) else dur_total
        t0 = max(b["start"], cursor + 0.06)
        disponible = max(0.4, sig - t0 - 0.1)
        d = len(clip) / SR
        factor = 1.0
        if d > disponible * 1.01:
            factor = min(max_acel, d / disponible)
            clip = acelerar(clip, factor)
            d = len(clip) / SR
        if d > disponible:  # todavia no entra: arrancar un poco antes si hay silencio previo
            margen = t0 - (cursor + 0.06)
            adelanto = max(0.0, min(margen, 0.4, d - disponible))
            t0 -= adelanto
        piezas = [(t0, clip)]
        span = min(b["end"], sig - 0.1) - t0  # lo que dura el habla original de esta frase
        if repartir and factor == 1.0 and span > 1.5 and d < span * 0.92:
            # 1) frenar un poco (como mucho min_tempo) si el ingles quedo mucho mas corto
            if d < span * 0.8 and min_tempo < 1.0:
                f2 = max(min_tempo, d / (span * 0.9))
                if f2 < 0.99:
                    clip = acelerar(clip, f2)
                    factor = f2
                    d = len(clip) / SR
            # 2) repartir las frases sobre el habla original
            a0, z0 = int(t0 * SR), int((t0 + span) * SR)
            piezas = distribuir(clip, voz_orig[a0:z0], t0, cfg, limite=t0 + disponible)
        fin = t0
        for ts, audio in piezas:
            i0 = int(ts * SR)
            pista[i0:i0 + len(audio)] += audio
            fin = max(fin, ts + len(audio) / SR)
        cursor = fin
        colocados.append({"id": b["id"], "t0": round(piezas[0][0], 3), "t1": round(cursor, 3),
                          "factor": round(factor, 3), "piezas": len(piezas),
                          "desfase": round(max(0.0, cursor - sig), 2)})
    largo = max(n_total, int(cursor * SR) + 1)
    if largo > n_total:
        log(f"  AVISO: la voz doblada termina {largo / SR - dur_total:.1f}s despues del final del video (se corta)")
    return pista[:n_total], colocados



# ---------------------------------------------------------------- modo "ajustar video": el audio manda, el video se acelera
def fps_video(video):
    import re
    r = __import__("subprocess").run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace")
    m = re.search(r"(\d+(?:\.\d+)?) fps", r.stderr)
    return float(m.group(1)) if m else 25.0


CODECS_VIDEO = (
    ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p"],  # placa NVIDIA
    ["-c:v", "libx264", "-crf", "21", "-preset", "medium", "-pix_fmt", "yuv420p"],
)


def ffmpeg_video_exe():
    return _ffmpegs_con_libass()[0]  # el de imageio trae libx264 y libass


def nivel_db(x, a, b):
    z = x[max(0, int(a * SR)):max(0, int(b * SR))]
    if len(z) == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(z ** 2)) + 1e-9))


def plan_ajuste(bloques, dclips, dur_total, fondo_mono, fps, cfg):
    """Divide el video en tramos. Tramos 'habla' (una frase + su pausa) se aceleran/frenan para durar lo que
    dura el ingles; tramos 'libre' (placa, musica, final) quedan a velocidad normal."""
    q = 1.0 / fps

    def rnd(t):
        return round(t / q) * q

    pre = 0.12
    vmax = float(cfg.get("video_vel_max", 1.6))
    vmin = float(cfg.get("video_vel_min", 0.85))
    segs = []
    a = rnd(max(0.0, bloques[0]["start"] - pre))
    if a > 0:
        segs.append({"a": 0.0, "b": a, "f": 1.0, "n": a, "tipo": "libre"})
    for i, b in enumerate(bloques):
        sig = bloques[i + 1]["start"] - pre if i + 1 < len(bloques) else None
        fin_habla = b["end"]
        corte = None
        if sig is None:
            corte = min(dur_total, rnd(fin_habla + 0.5))
        elif sig - fin_habla > 3.0 and nivel_db(fondo_mono, fin_habla + 0.4, sig - 0.2) > -45:
            corte = rnd(fin_habla + 0.4)  # pausa larga con musica/sonido: se deja a velocidad normal
        z = corte if corte is not None else rnd(sig)
        z = max(z, a + q)
        orig = z - a
        pausa = min(0.7, max(0.2, (z - fin_habla) * 0.4))
        objetivo = pre + dclips[i] + pausa
        f = min(vmax, max(vmin, orig / objetivo))
        n = max(q, rnd(orig / f))
        segs.append({"a": a, "b": z, "f": orig / n, "n": n, "tipo": "habla", "i": i})
        if corte is not None:
            fin_libre = rnd(sig) if sig is not None else dur_total
            if fin_libre - corte > q / 2:
                segs.append({"a": corte, "b": fin_libre, "f": 1.0, "n": fin_libre - corte, "tipo": "libre"})
            a = fin_libre
        else:
            a = z
    t = 0.0
    for sg in segs:
        sg["T"] = t
        t += sg["n"]
    return segs


def mapear(t, segs):
    for sg in segs:
        if t < sg["b"] or sg is segs[-1]:
            t = min(max(t, sg["a"]), sg["b"])
            return sg["T"] + (t - sg["a"]) / sg["f"]
    return segs[-1]["T"] + segs[-1]["n"]


def estirar(x, f, n_out):
    """Cambia la duracion de un audio (mono o estereo) por el factor f (>1 = mas corto) y ajusta a n_out muestras."""
    import soundfile as sf
    if abs(f - 1.0) > 0.005 and len(x) > SR // 20:
        with tempfile.TemporaryDirectory() as td:
            a_, b_ = Path(td) / "a.wav", Path(td) / "b.wav"
            sf.write(str(a_), x, SR, subtype="FLOAT")
            cadena, g = [], f
            while g > 2.0:
                cadena.append("atempo=2.0")
                g /= 2.0
            while g < 0.5:
                cadena.append("atempo=0.5")
                g /= 0.5
            cadena.append(f"atempo={g:.6f}")
            ffmpeg("-i", a_, "-filter:a", ",".join(cadena), "-c:a", "pcm_f32le", b_)
            x, _ = sf.read(str(b_), dtype="float32", always_2d=(x.ndim == 2))
    if len(x) >= n_out:
        return x[:n_out]
    pad = ((0, n_out - len(x)),) + (((0, 0),) if x.ndim == 2 else ())
    return np.pad(x, pad)


def leer_srt(path):
    import re
    txt = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    cues = []
    for bl in re.split(r"\n\s*\n", txt.strip()):
        partes = bl.strip().split("\n")
        if len(partes) >= 3 and "-->" in partes[1]:
            def seg(t):
                h, m, rest = t.strip().split(":")
                s_, ms = rest.split(",")
                return int(h) * 3600 + int(m) * 60 + int(s_) + int(ms) / 1000
            a, z = partes[1].split("-->")
            cues.append((seg(a), seg(z), "\n".join(partes[2:])))
    return cues


def video_ajustado(video, segs, mezcla, srt, destino, fps, sub_cod="es"):
    partes, etiquetas = [f"[0:v]split={len(segs)}" + "".join(f"[s{k}]" for k in range(len(segs)))], []
    for k, sg in enumerate(segs):
        partes.append(f"[s{k}]trim=start={sg['a']:.4f}:end={sg['b']:.4f},setpts=(PTS-STARTPTS)/{sg['f']:.6f}[v{k}]")
        etiquetas.append(f"[v{k}]")
    fc = ";".join(partes) + ";" + "".join(etiquetas) + f"concat=n={len(segs)}:v=1:a=0,fps={fps:g}[vout]"
    tmp = destino.with_suffix(".tmp.mp4")
    ultimo = None
    for codec in CODECS_VIDEO:
        try:
            correr([ffmpeg_video_exe(), "-hide_banner", "-nostdin", "-y", "-i", video, "-i", mezcla, "-i", srt,
                    "-filter_complex", fc, "-map", "[vout]", "-map", "1:a:0", "-map", "2:0", *codec,
                    "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
                    "-metadata:s:a:0", f"language={ISO3}", "-metadata:s:s:0", f"language={_iso3(sub_cod)}",
                    "-metadata:s:s:0", f"title={_nombre_idioma(sub_cod)}", "-disposition:s:0", "default",
                    "-movflags", "+faststart", tmp])
            tmp.replace(destino)
            return
        except RuntimeError as e:
            ultimo = e
    raise ultimo


def doblar_ajustando_video(d, w, cfg, es, bloques, clips, informe, video, out_mp4, quemado, srt_es, sal, stem,
                           audio=False, src="es"):
    """Modo 'ajustar video': la voz doblada va a velocidad natural y el video (o el fondo, si es un audio) se
    acelera o frena por tramos para acompanarla."""
    import soundfile as sf
    voz_orig, _ = sf.read(str(d / "voz.flac"), dtype="float32")
    if voz_orig.ndim > 1:
        voz_orig = voz_orig.mean(1)
    fondo, _ = sf.read(str(d / "fondo.flac"), dtype="float32", always_2d=True)
    dur_total = es["duracion"]
    if audio:
        # sin imagen: la linea de tiempo se adapta libremente a la voz nueva (el fondo se estira o encoge)
        fps = 100.0
        cfg = dict(cfg, video_vel_max=cfg.get("audio_vel_max", 3.0), video_vel_min=cfg.get("audio_vel_min", 0.5))
    else:
        fps = fps_video(video)
    segs = plan_ajuste(bloques, [len(c) / SR for c in clips], dur_total, fondo.mean(1), fps, cfg)
    total = segs[-1]["T"] + segs[-1]["n"]
    N = int(round(total * SR))
    progreso(d.name, 3, 82, "mezclando la voz con la música")
    log(f"  {'Audio' if audio else 'Video'} ajustado al {cfg.get('idioma_nombre', 'ingles')}: {dur_total:.0f}s -> {total:.0f}s "
        f"(velocidad media {dur_total / total:.2f}x, max {max(sg['f'] for sg in segs):.2f}x)")

    # voz en ingles, a velocidad natural, una frase detras de otra
    niveles = [rms_activo(voz_orig[int(b["start"] * SR):int(b["end"] * SR)]) for b in bloques]
    med = float(np.median([n_ for n_ in niveles if n_ > 0]) or 0.1)
    pista = np.zeros(N + SR * 5, np.float32)
    habla = {sg["i"]: sg for sg in segs if sg["tipo"] == "habla"}
    cursor, colocados = 0.0, []
    for i, (b, clip) in enumerate(zip(bloques, clips)):
        sg = habla[i]
        r = rms_activo(clip)
        if r > 0:
            clip = clip * (float(np.clip(niveles[i] or med, med * 0.5, med * 2.0)) / r)
        t0 = max(mapear(b["start"], segs), cursor + 0.08)
        fin_tramo = sg["T"] + sg["n"]
        factor = 1.0
        disp = fin_tramo - t0 + 0.3
        if len(clip) / SR > disp and disp > 0.5:  # solo si el video ya no se puede frenar mas
            factor = min(1.15, (len(clip) / SR) / disp)
            clip = acelerar(clip, factor)
        i0 = int(t0 * SR)
        pista[i0:i0 + len(clip)] += clip
        cursor = t0 + len(clip) / SR
        colocados.append({"id": b["id"], "t0": round(t0, 3), "t1": round(cursor, 3), "factor": round(factor, 3),
                          "vel_video": round(sg["f"], 3), "desfase": round(max(0.0, cursor - fin_tramo), 2)})
    pista = pista[:N]

    # fondo (musica/ambiente) y voz original de los tramos libres, con la misma linea de tiempo
    fondo_n = np.zeros((N, fondo.shape[1]), np.float32)
    voz_libre = np.zeros(N, np.float32)
    for sg in segs:
        i0, n_out = int(round(sg["T"] * SR)), int(round(sg["n"] * SR))
        n_out = min(n_out, N - i0)
        if n_out <= 0:
            continue
        trozo = fondo[int(sg["a"] * SR):int(sg["b"] * SR)]
        fondo_n[i0:i0 + n_out] = estirar(trozo, sg["f"], n_out)
        if sg["tipo"] == "libre" and cfg.get("conservar_voz_original", True):
            voz_libre[i0:i0 + n_out] = estirar(voz_orig[int(sg["a"] * SR):int(sg["b"] * SR)], 1.0, n_out)

    r_orig, r_new = rms_activo(voz_orig), rms_activo(pista)
    if r_new > 0:
        pista *= r_orig / r_new
    fondo_n = fondo_n * (10 ** (float(cfg.get("fondo_ajuste_db", 0)) / 20))
    fondo_n = ducking(fondo_n, pista, float(cfg.get("atenuar_fondo_db", 6)))
    mezcla = fondo_n + pista[:, None] + voz_libre[:, None]
    pico = np.max(np.abs(mezcla))
    if pico > 0.99:
        mezcla *= 0.99 / pico
    sf.write(str(w / "voz_doblada.wav"), pista, SR, subtype="PCM_16")
    sf.write(str(w / "mezcla_pre.wav"), mezcla, SR, subtype="FLOAT")
    lufs_in = loudnorm(w / "mezcla_pre.wav", w / "mezcla.wav", float(cfg.get("loudness_lufs", -14)))
    (w / "mezcla_pre.wav").unlink(missing_ok=True)

    # subtitulos en la nueva linea de tiempo
    cues_es = []
    for a, z, txt in leer_srt(srt_es):
        na, nz = mapear(a, segs), mapear(z, segs)
        cues_es.append([na, max(nz, na + 0.8), txt])
    for k in range(len(cues_es) - 1):
        cues_es[k][1] = min(cues_es[k][1], cues_es[k + 1][0] - 0.04)
    srt_es_aj = w / "es_ajustado.srt"
    escribir_srt(srt_es_aj, [tuple(c) for c in cues_es])
    cues_en = []
    for b, c in zip(bloques, colocados):
        cues_en += texto_a_cues(b["en"], c["t0"], c["t1"])
    srt_en = w / "traduccion.srt"
    escribir_srt(srt_en, cues_en)

    usar_trad, sub_cod = idioma_subs(src, cfg.get("idioma_codigo", "en"))
    srt_sub = srt_en if usar_trad else srt_es_aj
    (sal / f"{stem} [{src.upper()}].srt").write_bytes(srt_es_aj.read_bytes())  # lo que se dice en el original
    (sal / f"{stem} [{SUFIJO}].srt").write_bytes(srt_en.read_bytes())          # lo que dice el doblaje
    if audio:
        log("  Armando el audio final...")
        progreso(d.name, 3, 90, "armando el audio")
        out_mp4 = exportar_audio(w / "mezcla.wav", out_mp4, titulo=f"{stem} [{SUFIJO}]")
    else:
        log("  Armando video final (re-codificando la imagen)...")
        progreso(d.name, 3, 88, "armando el video")
        video_ajustado(video, segs, w / "mezcla.wav", srt_sub, out_mp4, fps, sub_cod)
        if cfg.get("quemar_subtitulos"):
            try:
                progreso(d.name, 3, 95, "grabando los subtítulos en la imagen")
                arriba = [(mapear(a, segs), mapear(z, segs)) for a, z in detectar_zocalos(video, d)]
                quemar_subtitulos(out_mp4, w / "mezcla.wav", srt_sub, quemado, w, arriba=arriba)
            except Exception:
                log_error("grabar subtitulos (el video con subtitulos seleccionables igual quedo bien)")

    flojos = [x for x in informe if (x.get("wer") or 0) > float(cfg.get("wer_max", 0.2)) or x.get("fuga")]
    desfasados = [c for c in colocados if c["desfase"] > 0.3]
    escribir_json(w / "informe.json", {"modo": "ajustar_audio" if audio else "ajustar_video", "lufs_entrada": lufs_in,
                                       "salida": out_mp4.name, "duracion_original": dur_total,
                                       "duracion_nueva": total, "tramos": segs, "bloques": informe,
                                       "colocados": colocados, "revisar_wer": [x["id"] for x in flojos],
                                       "acelerados": [(c["id"], c["factor"]) for c in colocados if c["factor"] > 1],
                                       "desfasados": [(c["id"], c["desfase"]) for c in desfasados]})
    log(f"[{d.name}] LISTO -> {out_mp4.name} | {dur_total:.0f}s -> {total:.0f}s | bloques a revisar: "
        f"{[x['id'] for x in flojos] or 'ninguno'} | desfasados: {len(desfasados)}")


def _anotar_voz(w, ref_hash):
    p = w / "informe.json"
    if p.exists():
        inf = leer_json(p)
        inf["ref_hash"] = ref_hash
        escribir_json(p, inf)


def doblar(d, cfg, motor):
    import soundfile as sf

    es = leer_json(d / "es.json")
    trad = archivo_traduccion(d, cfg)
    en = leer_json(trad)
    w = carpeta_idioma(d, cfg)
    ref_wav, ref_txt = referencia_voz(d, cfg)
    video = Path(cfg["carpeta_videos"]) / es["video"]
    audio = es_audio(video)
    src = idioma_origen(es, cfg)
    cod = cfg.get("idioma_codigo", "en")
    stem = video.stem
    sal = carpeta_salida(cfg)
    srt_es = d / "es.srt" if (d / "es.srt").exists() else d / "es_auto.srt"
    usar_trad, sub_cod = idioma_subs(src, cod)
    if audio:
        out_mp4 = sal / f"{stem} [{SUFIJO}].mp3"
        if not out_mp4.exists() and out_mp4.with_suffix(".m4a").exists():
            out_mp4 = out_mp4.with_suffix(".m4a")
    else:
        out_mp4 = sal / f"{stem} [{SUFIJO}].mp4"
    quemado = sal / f"{stem} [{SUFIJO}] subs {sub_cod.upper()}.mp4"
    fuentes = [trad, srt_es, ref_wav]
    ref_hash = hashlib.md5(ref_wav.read_bytes() + ref_txt.encode()).hexdigest()[:10]
    previo = leer_json(w / "informe.json") if (w / "informe.json").exists() else {}
    # si se cambio la voz (muestra subida <-> voz del video) hay que volver a doblar aunque nada sea mas nuevo
    misma_voz = previo.get("ref_hash") == ref_hash or (previo.get("ref_hash") is None and not cfg.get("voz_muestra"))
    if out_mp4.exists() and not FORZAR and misma_voz and all(out_mp4.stat().st_mtime > f.stat().st_mtime for f in fuentes):
        if cfg.get("quemar_subtitulos") and not cfg.get("ajustar_video") and not audio and (w / "mezcla.wav").exists() and (
                not quemado.exists() or quemado.stat().st_mtime < out_mp4.stat().st_mtime):
            log(f"[{d.name}] doblaje al dia; falta la version con subtitulos grabados")
            try:
                quemar_subtitulos(video, w / "mezcla.wav", w / "traduccion.srt" if usar_trad else srt_es, quemado, w)
                log(f"[{d.name}] LISTO -> {quemado.name}")
            except Exception:
                log_error("grabar subtitulos")
        else:
            log(f"[{d.name}] ya doblado y sin cambios, salteo")
        progreso(d.name, 3, 100, "ya estaba doblado")
        return
    log(f"[{d.name}] DOBLANDO: {video.name}" + (f" (audio en {ORIGENES.get(src, (src, src))[1]})" if src != "es" else ""))
    bloques = en["bloques"]

    progreso(d.name, 3, 1, "cargando la voz")
    clips, informe = generar_bloques(motor, w, bloques, cfg, ref_wav, ref_txt,
                                     avance=lambda i, n: progreso(d.name, 3, 2 + 78 * i / max(n, 1),
                                                                  f"generando la voz: frase {i + 1} de {n}"))
    if cfg.get("ajustar_video") or audio:
        doblar_ajustando_video(d, w, cfg, es, bloques, clips, informe, video, out_mp4, quemado, srt_es, sal, stem,
                               audio=audio, src=src)
        _anotar_voz(w, ref_hash)
        progreso(d.name, 3, 100, "listo")
        return

    voz_orig, _ = sf.read(str(d / "voz.flac"), dtype="float32")
    if voz_orig.ndim > 1:
        voz_orig = voz_orig.mean(1)
    fondo, sr_f = sf.read(str(d / "fondo.flac"), dtype="float32", always_2d=True)
    dur_total = es["duracion"]
    pista, colocados = ensamblar(bloques, clips, voz_orig, dur_total, cfg)

    # balance voz/fondo igual que el original
    r_orig, r_new = rms_activo(voz_orig), rms_activo(pista)
    if r_new > 0:
        pista *= r_orig / r_new
    n = len(pista)
    if len(fondo) < n:
        fondo = np.pad(fondo, ((0, n - len(fondo)), (0, 0)))
    fondo = fondo[:n] * (10 ** (float(cfg.get("fondo_ajuste_db", 0)) / 20))
    fondo = ducking(fondo, pista, float(cfg.get("atenuar_fondo_db", 6)))
    mezcla = fondo + pista[:, None]

    # fuera de las frases traducidas se deja la voz original (canciones, cantantes, risas, etc.)
    if cfg.get("conservar_voz_original", True):
        fs = 100  # mascara a 100 Hz y despues interpolada
        nm = int(n / SR * fs) + 1
        m = np.ones(nm, np.float32)
        for b_, c in zip(bloques, colocados):
            for a, z in ((b_["start"] - 0.3, b_["end"] + 0.3), (c["t0"] - 0.15, c["t1"] + 0.15)):
                m[max(0, int(a * fs)):max(0, int(z * fs))] = 0
        m = np.clip(suavizar(m, 15), 0, 1)
        mask = np.interp(np.arange(n) / SR, np.arange(nm) / fs, m).astype(np.float32)
        vo = voz_orig[:n] if len(voz_orig) >= n else np.pad(voz_orig, (0, n - len(voz_orig)))
        resto = vo * mask
        hop = int(0.25 * SR)
        k = len(resto) // hop
        e = np.sqrt(np.mean(resto[:k * hop].reshape(k, hop) ** 2, axis=1))
        seg_vivos = float((e > 10 ** (-40 / 20)).sum()) * 0.25
        if seg_vivos > 0:
            log(f"  Conservo {seg_vivos:.1f}s de voz original fuera de las frases traducidas (canciones, etc.)")
        mezcla = mezcla + resto[:, None]
    pico = np.max(np.abs(mezcla))
    if pico > 0.99:
        mezcla *= 0.99 / pico

    sf.write(str(w / "voz_doblada.wav"), pista, SR, subtype="PCM_16")
    sf.write(str(w / "mezcla_pre.wav"), mezcla, SR, subtype="FLOAT")
    lufs_in = loudnorm(w / "mezcla_pre.wav", w / "mezcla.wav", float(cfg.get("loudness_lufs", -14)))
    (w / "mezcla_pre.wav").unlink(missing_ok=True)

    # subtitulos ingles a partir de donde quedo cada frase
    cues_en = []
    for b, c in zip(bloques, colocados):
        cues_en += texto_a_cues(b["en"], c["t0"], c["t1"])
    srt_en = w / "traduccion.srt"
    escribir_srt(srt_en, cues_en)

    log("  Armando video final...")
    srt_sub = srt_en if usar_trad else srt_es
    tmp = out_mp4.with_suffix(".tmp.mp4")
    ffmpeg("-i", video, "-i", w / "mezcla.wav", "-i", srt_sub,
           "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
           "-metadata:s:a:0", f"language={ISO3}", "-metadata:s:s:0", f"language={_iso3(sub_cod)}",
           "-metadata:s:s:0", f"title={_nombre_idioma(sub_cod)}", "-disposition:s:0", "default",
           "-movflags", "+faststart", tmp)
    tmp.replace(out_mp4)
    (sal / f"{stem} [{src.upper()}].srt").write_bytes(srt_es.read_bytes())
    (sal / f"{stem} [{SUFIJO}].srt").write_bytes(srt_en.read_bytes())

    if cfg.get("quemar_subtitulos"):
        try:
            quemar_subtitulos(video, w / "mezcla.wav", srt_sub, quemado, w)
        except Exception:
            log_error("grabar subtitulos (el video con subtitulos seleccionables igual quedo bien)")

    flojos = [x for x in informe if (x.get("wer") or 0) > float(cfg.get("wer_max", 0.2)) or x.get("fuga")]
    acelerados = [c for c in colocados if c["factor"] > 1.0]
    frenados = [c for c in colocados if c["factor"] < 1.0]
    desfasados = [c for c in colocados if c["desfase"] > 0.3]
    escribir_json(w / "informe.json", {"lufs_entrada": lufs_in, "bloques": informe, "colocados": colocados,
                                       "revisar_wer": [x["id"] for x in flojos],
                                       "acelerados": [(c["id"], c["factor"]) for c in acelerados],
                                       "desfasados": [(c["id"], c["desfase"]) for c in desfasados]})
    log(f"[{d.name}] LISTO -> {out_mp4.name} | bloques a revisar: {[x['id'] for x in flojos] or 'ninguno'}"
        f" | acelerados: {len(acelerados)} | desfasados: {len(desfasados)}")
    _anotar_voz(w, ref_hash)
    progreso(d.name, 3, 100, "listo")


def main(cfg):
    global SUFIJO, ISO3
    SUFIJO, ISO3 = cfg.get("sufijo_salida", "EN"), cfg.get("idioma_iso3", "eng")
    proc = carpeta_proceso(cfg)
    caps = sorted(p for p in proc.iterdir() if p.is_dir() and (p / "es.json").exists())
    if SOLO:
        caps = [p for p in caps if p.name in SOLO]
    pendientes = [p for p in caps if archivo_traduccion(p, cfg).exists()]
    sin_trad = [p.name for p in caps if not archivo_traduccion(p, cfg).exists()]
    log(f"Idioma: {cfg['idioma_nombre']} | voz: {'muestra subida' if cfg.get('voz_muestra') else 'la del video'}")
    if sin_trad:
        log(f"Sin traduccion al {cfg['idioma_nombre']} todavia (falta el Paso 2): {', '.join(sin_trad)}")
    motor = Motor(cfg)
    errores = 0
    for d in pendientes:
        try:
            doblar(d, cfg, motor)
        except Exception:
            errores += 1
            log_error(d.name)
    log(f"Paso 3 terminado. Errores: {errores}")
    if errores:
        sys.exit(1)


if __name__ == "__main__":
    ejecutar_principal("PASO 3 (doblar)", main)
