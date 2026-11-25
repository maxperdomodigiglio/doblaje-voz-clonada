"""Partes cantadas: detectarlas (Paso 1), preparar la letra (Paso 2) y cantarla en el otro idioma (Paso 3).

- Deteccion: un clasificador de sonidos (AST entrenado con AudioSet) escucha la voz separada en ventanas de
  4 s y dice si es habla o canto. Los tramos cantados se vuelven a transcribir en su propio idioma.
- Canto: YingMusic-Singer-Plus vuelve a cantar cada trozo con la letra traducida, siguiendo la melodia de la
  voz original y con su timbre. Corre en su propio entorno (env_canto) via canto/canto_motor.py.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from comun import APP_DIR, ORIGENES, escribir_json, leer_json, log

VERSION_DET = 1
AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
ETQ_CANTO = {"Singing", "Choir", "Yodeling", "Chant", "Mantra", "Male singing", "Female singing", "Child singing",
             "Synthetic singing", "Rapping", "Humming", "A capella", "Vocal music"}
ETQ_HABLA = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
             "Conversation", "Narration, monologue", "Speech synthesizer", "Whispering"}
REPO = APP_DIR / "canto" / "YingMusic-Singer-Plus"
MOTOR = APP_DIR / "canto" / "canto_motor.py"
SR16 = 16000


def python_canto():
    base = APP_DIR / "env_canto"
    for p in (base / "Scripts" / "python.exe", base / "bin" / "python"):
        if p.exists():
            return p
    return None


def motor_instalado():
    return python_canto() is not None and (REPO / "src" / "YingMusicSinger").is_dir()


# ---------------------------------------------------------------- deteccion
def _db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if len(x) else -120.0


def puntajes(y, cfg, avance=None):
    """Ventanas [t0, t1, canto, habla, dB] sobre la voz separada (16 kHz mono): 4 s cada 1 s."""
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = str(APP_DIR / "modelos" / "hf")
    nombre = cfg.get("canto_detector", AST)
    fe = ASTFeatureExtractor.from_pretrained(nombre, cache_dir=cache)
    model = ASTForAudioClassification.from_pretrained(nombre, cache_dir=cache).to(dev).eval()
    etq = model.config.id2label
    ic = [int(i) for i, l in etq.items() if l in ETQ_CANTO]
    ih = [int(i) for i, l in etq.items() if l in ETQ_HABLA]
    win, hop = float(cfg.get("canto_ventana", 4.0)), float(cfg.get("canto_paso", 1.0))
    dur = len(y) / SR16
    ts = list(np.arange(0.0, max(dur - win * 0.5, 0.01), hop))
    vent, lote = [], []

    def procesar():
        trozos = [y[int(a * SR16):int((a + win) * SR16)] for a in lote]
        x = fe([t.astype(np.float32) for t in trozos], sampling_rate=SR16, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(model(**{k: v.to(dev) for k, v in x.items()}).logits).cpu().numpy()
        for a, t, pp in zip(lote, trozos, p):
            vent.append([round(a, 2), round(min(dur, a + win), 2), round(float(pp[ic].max()), 3),
                         round(float(pp[ih].max()), 3), round(float(_db(t)), 1)])
        lote.clear()

    for k, a in enumerate(ts):
        lote.append(a)
        if len(lote) == 16:
            procesar()
            if avance:
                avance(k + 1, len(ts))
    if lote:
        procesar()
    del model
    return vent


def tramos_desde(vent, y, cfg):
    """Une las ventanas cantadas en tramos y ajusta los bordes a donde empieza/termina la voz."""
    umbral = float(cfg.get("canto_umbral", 0.25))
    razon = float(cfg.get("canto_razon", 1.0))
    min_db = float(cfg.get("canto_min_db", -42))
    marca = [v[4] > min_db and v[2] >= umbral and v[2] >= razon * v[3] for v in vent]
    # suaviza: una ventana suelta distinta de sus dos vecinas se corrige
    for i in range(1, len(marca) - 1):
        if marca[i - 1] == marca[i + 1] != marca[i]:
            marca[i] = marca[i - 1]
    tramos = []
    for v, m in zip(vent, marca):
        if not m:
            continue
        c, h = (v[0] + v[1]) / 2, float(cfg.get("canto_paso", 1.0)) / 2  # cada ventana responde por su centro
        a, b = c - h, c + h
        if tramos and a - tramos[-1][1] <= float(cfg.get("canto_unir_seg", 2.0)):
            tramos[-1][1] = max(tramos[-1][1], b)
        else:
            tramos.append([a, b])
    # bordes: se recorta el silencio de las puntas y se busca el comienzo real de la voz
    hop = int(0.05 * SR16)
    n = len(y) // hop
    env = np.array([_db(y[i * hop:(i + 1) * hop]) for i in range(n)]) if n else np.zeros(0)
    activo = env > min_db - 4
    out = []
    for a, b in tramos:
        ia, ib = int(a / 0.05), min(n, int(b / 0.05))
        while ia < ib and not activo[ia]:
            ia += 1
        while ib > ia and not activo[ib - 1]:
            ib -= 1
        k = 0  # hacia atras, mientras haya voz (hasta 1,5 s), para no cortar la primera palabra
        while ia > 0 and activo[ia - 1] and k < 30:
            ia -= 1
            k += 1
        k = 0  # y hacia adelante (hasta 0,5 s), para no cortar la ultima nota
        while ib < n and activo[ib] and k < 10:
            ib += 1
            k += 1
        a2, b2 = ia * 0.05, ib * 0.05
        if b2 - a2 >= float(cfg.get("canto_min_seg", 4.0)):
            out.append([round(a2, 2), round(b2, 2)])
    return out


def lineas_de(segmentos, max_dur=7.0):
    """Segmentos de Whisper (con palabras) -> lineas de la letra, partiendo las largas en su pausa mas grande."""
    lineas = []

    def agregar(ws):
        if ws:
            lineas.append({"t0": round(ws[0]["s"], 2), "t1": round(ws[-1]["e"], 2),
                           "texto": " ".join(w["w"] for w in ws).strip(), "palabras": ws})

    def partir(ws):
        if len(ws) > 2 and ws[-1]["e"] - ws[0]["s"] > max_dur:
            # la pausa mas larga (mejor si no deja una palabra sola en una punta)
            gaps = [(ws[i + 1]["s"] - ws[i]["e"] - (0.3 if i in (0, len(ws) - 2) else 0.0), i)
                    for i in range(len(ws) - 1)]
            _, i = max(gaps)
            partir(ws[:i + 1])
            partir(ws[i + 1:])
        else:
            agregar(ws)

    for sg in segmentos:
        ws = [{"w": w["w"], "s": round(w["s"], 2), "e": round(w["e"], 2)} for w in sg.get("words", []) if w["w"].strip()]
        if ws:
            partir(ws)
        elif sg.get("text", "").strip():
            lineas.append({"t0": round(sg["start"], 2), "t1": round(sg["end"], 2), "texto": sg["text"].strip(),
                           "palabras": []})
    return lineas


def trozos_de(tramo, max_dur=12.0, gap_max=1.6):
    """Agrupa las lineas de un tramo en trozos de hasta ~12 s para cantarlos de a uno."""
    trozos, cur = [], []
    for ln in tramo["lineas"]:
        if cur and (ln["t1"] - cur[0]["t0"] > max_dur or ln["t0"] - cur[-1]["t1"] > gap_max):
            trozos.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        trozos.append(cur)
    out = []
    for k, ls in enumerate(trozos):
        a = max(tramo["t0"], ls[0]["t0"] - 0.15)
        b = min(tramo["t1"], ls[-1]["t1"] + 0.3)
        if out:
            a = max(a, out[-1]["t1"])
        out.append({"k": k, "t0": round(a, 2), "t1": round(max(b, a + 0.5), 2), "lineas": ls})
    return out


# ---------------------------------------------------------------- silabas
_VOC = {
    "es": "aeiouáéíóúü", "it": "aeiouàèéìíòóùú", "pt": "aeiouáéíóúâêôãõàü", "fr": "aeiouyéèêëàâîïôûùü",
    "de": "aeiouyäöü", "en": "aeiouy",
}
_RU = "аеёиоуыэюя"


def silabas(texto, cod):
    """Cuenta aproximada de silabas (grupos de vocales; en ruso, cada vocal)."""
    total = 0
    for p in re.findall(r"[^\W\d_]+", texto.lower()):
        if cod == "ru" or re.search(r"[а-яё]", p):
            total += sum(ch in _RU for ch in p)  # 'с', 'в', 'к' no suman silaba
            continue
        voc = _VOC.get(cod, _VOC["es"])
        grupos = re.findall(f"[{voc}]+", p)
        n = len(grupos)
        if cod == "en" and n > 1 and p.endswith("e") and not p.endswith(("le", "ee", "ye")):
            n -= 1
        if cod == "fr" and n > 1 and re.search(r"e[s]?$", p):
            n -= 1
        total += max(1, n)
    return total


# ---------------------------------------------------------------- afinacion y sonido del canto (Paso 3)
def _ajustar_largo(x, n):
    return np.pad(x, (0, max(0, n - len(x))))[:n]


def f0_de(y, sr, periodo=5.0):
    """Altura (Hz) cada 5 ms con WORLD (0 donde no hay voz)."""
    import pyworld as pw
    x = np.ascontiguousarray(y, dtype=np.float64)
    f0, t = pw.dio(x, sr, f0_floor=60.0, f0_ceil=1100.0, frame_period=periodo)
    return pw.stonemask(x, f0, t, sr), t


def desafinacion(gen, orig, sr):
    """Cuanto se aparta el canto generado de la melodia original: (cents medianos, octavas de diferencia)."""
    import librosa
    n = len(gen)
    g16 = librosa.resample(gen, orig_sr=sr, target_sr=16000)
    o16 = librosa.resample(_ajustar_largo(orig, n), orig_sr=sr, target_sr=16000)
    fg, _ = f0_de(g16, 16000)
    fo, _ = f0_de(o16, 16000)
    m = min(len(fg), len(fo))
    fg, fo = fg[:m], fo[:m]
    ambos = (fg > 0) & (fo > 0)
    if ambos.sum() < 20:
        return None, 0
    dif = 1200 * np.log2(fg[ambos] / fo[ambos])
    octv = int(np.round(np.median(dif) / 1200))
    return float(np.median(np.abs(dif - 1200 * octv))), octv


def afinar(gen, orig, sr, fuerza=1.0):
    """Autotune guiado: lleva la altura del canto generado a la de la voz original (la misma melodia, con su
    vibrato), en la octava en que canta la voz nueva. Conserva el timbre y las vocales (WORLD).
    Devuelve (audio, cents_antes, cents_despues)."""
    import pyworld as pw
    n = len(gen)
    x = np.ascontiguousarray(gen, dtype=np.float64)
    f0g, t = pw.dio(x, sr, f0_floor=60.0, f0_ceil=1100.0, frame_period=5.0)
    f0g = pw.stonemask(x, f0g, t, sr)
    fo, _ = f0_de(_ajustar_largo(orig, n), sr)
    m = min(len(f0g), len(fo))
    fo = np.pad(fo[:m], (0, len(f0g) - m))
    ambos = (f0g > 0) & (fo > 0)
    if ambos.sum() < 20:
        return gen, None, None
    dif = 1200 * np.log2(f0g[ambos] / fo[ambos])
    octv = np.round(np.median(dif) / 1200)
    meta = fo * (2.0 ** octv)
    antes = float(np.median(np.abs(dif - 1200 * octv)))
    nuevo = f0g.copy()
    lg, lm = np.log2(f0g[ambos]), np.log2(meta[ambos])
    nuevo[ambos] = 2.0 ** (lg + fuerza * (lm - lg))
    sp = pw.cheaptrick(x, f0g, t, sr)
    ap = pw.d4c(x, f0g, t, sr)
    y = pw.synthesize(nuevo, sp, ap, sr, frame_period=5.0).astype(np.float32)
    despues = float(np.median(np.abs(1200 * np.log2(nuevo[ambos] / meta[ambos]))))
    return _ajustar_largo(y, n), antes, despues


def sonido_voz(y, sr, cfg):
    """Cadena de voz cantada: limpia graves, da presencia (se entiende mejor la letra), comprime suave y le pone
    un poco de sala para que no suene 'pegada' sobre la musica."""
    import tempfile

    import soundfile as sf

    from comun import ffmpeg
    eco = float(cfg.get("canto_sala", 0.12))
    cadena = ["highpass=f=90", "equalizer=f=250:t=q:w=1.0:g=-2", "equalizer=f=3200:t=q:w=1.2:g=3",
              "acompressor=threshold=0.125:ratio=3:attack=8:release=120:makeup=1.4"]
    if eco > 0:
        cadena.append(f"aecho=1.0:1.0:45|85|130:{eco:.2f}|{eco * 0.7:.2f}|{eco * 0.45:.2f}")
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(str(a), y, sr, subtype="FLOAT")
        ffmpeg("-i", a, "-filter:a", ",".join(cadena), "-c:a", "pcm_f32le", b)
        z, _ = sf.read(str(b), dtype="float32")
    return _ajustar_largo(z, len(y))


# ---------------------------------------------------------------- canto (Paso 3)
def huella(*partes):
    return hashlib.md5(json.dumps(partes, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def correr_motor(trabajo_json, log_cb=None):
    """Corre canto_motor.py en env_canto. Devuelve {id: resultado}."""
    py = python_canto()
    if py is None:
        raise RuntimeError("El canto no esta instalado (en Pinokio: 'Instalar canto')")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    p = subprocess.Popen([str(py), str(MOTOR), str(trabajo_json)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
    ultimas = []
    for linea in p.stdout:
        linea = linea.rstrip()
        if linea.startswith("@@P "):
            print(linea, flush=True)  # el progreso pasa derecho a la interfaz
            continue
        if not linea.strip() or "Warning" in linea or "warnings.warn" in linea or "autocast" in linea:
            continue
        ultimas = (ultimas + [linea])[-30:]
        if linea.startswith("[canto]") or "Error" in linea or "error" in linea:
            log("  " + linea)
    p.wait()
    res = Path(trabajo_json).with_name(Path(trabajo_json).stem + "_resultado.json")
    if not res.exists():
        raise RuntimeError("el motor de canto se corto:\n" + "\n".join(ultimas[-12:]))
    return {r["id"]: r for r in leer_json(res)["items"]}
