"""Utilidades compartidas del Doblaje Gregorio (Pinokio)."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import unicodedata
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
EXT_VIDEO = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
# codigo -> (nombre para Qwen3-TTS, codigo ISO 639-2 del audio, sufijo de los archivos, nombre en castellano)
IDIOMAS = {
    "en": ("English", "eng", "EN", "Inglés"),
    "de": ("German", "deu", "DE", "Alemán"),
    "ru": ("Russian", "rus", "RU", "Ruso"),
}


def archivo_traduccion(d, cfg):
    """en.json para ingles (como siempre), de.json / ru.json para los otros idiomas."""
    return Path(d) / f"{cfg.get('idioma_codigo', 'en')}.json"


def carpeta_idioma(d, cfg):
    """Archivos de trabajo del doblaje: en la carpeta del capitulo para ingles, en una subcarpeta para los demas."""
    cod = cfg.get("idioma_codigo", "en")
    w = Path(d) if cod == "en" else Path(d) / cod
    w.mkdir(parents=True, exist_ok=True)
    return w


def referencia_voz(d, cfg):
    """(ref.wav, texto) de la voz a clonar: la muestra subida en la interfaz, o la elegida del propio video."""
    m = cfg.get("voz_muestra")
    if m and (Path(m) / "ref.wav").exists() and (Path(m) / "ref.txt").exists():
        base = Path(m)
    else:
        base = Path(d)
    return base / "ref.wav", (base / "ref.txt").read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------- config / rutas
def cargar_config():
    cfg = json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))
    # --config=otro.json pisa algunas claves (p.ej. otro idioma / otra carpeta) sin tocar config.json
    for a in sys.argv[1:]:
        if a.startswith("--config="):
            extra = Path(a.split("=", 1)[1])
            if not extra.is_absolute():
                extra = APP_DIR / extra
            cfg.update(json.loads(extra.read_text(encoding="utf-8")))
    # idioma de destino: con el codigo (en/de/ru) se completa el resto
    cod = cfg.get("idioma_codigo") or {v[0].lower(): k for k, v in IDIOMAS.items()}.get(
        str(cfg.get("idioma_destino", "English")).lower(), "en")
    nombre, iso3, sufijo, _ = IDIOMAS.get(cod, IDIOMAS["en"])
    cfg.update(idioma_codigo=cod, idioma_destino=nombre, idioma_iso3=iso3, sufijo_salida=sufijo,
               idioma_nombre=IDIOMAS.get(cod, IDIOMAS['en'])[3].lower())
    cfg["carpeta_videos"] = os.path.expandvars(os.path.expanduser(cfg["carpeta_videos"]))
    return cfg


def carpeta_videos(cfg):
    return Path(cfg["carpeta_videos"])


def carpeta_proceso(cfg):
    p = carpeta_videos(cfg) / "_proceso"
    p.mkdir(parents=True, exist_ok=True)
    return p


def carpeta_salida(cfg):
    p = carpeta_videos(cfg) / "salida"
    p.mkdir(parents=True, exist_ok=True)
    return p


def listar_videos(cfg):
    base = carpeta_videos(cfg)
    return sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() in EXT_VIDEO)


def slug_de(nombre):
    m = re.search(r"cap[ií]tulo\s*0*(\d+)", nombre, re.I)
    if m:
        return f"cap{int(m.group(1)):02d}"
    s = unicodedata.normalize("NFKD", Path(nombre).stem).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return (s[:40] or "video")


def leer_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def escribir_json(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- log
class Log:
    archivo = None

    @classmethod
    def iniciar(cls, archivo):
        cls.archivo = Path(archivo)
        cls.archivo.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def __call__(cls, *msg):
        pass


def log(*msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(m) for m in msg)
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)
    if Log.archivo:
        with open(Log.archivo, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def progreso(cap, paso, pct, detalle=""):
    """Avance para la interfaz (no va al log): una linea '@@P {json}' por la salida estandar."""
    try:
        print("@@P " + json.dumps({"cap": str(cap), "paso": paso, "pct": round(max(0.0, min(100.0, float(pct))), 1),
                                   "det": detalle}, ensure_ascii=False), flush=True)
    except Exception:
        pass


def log_error(contexto):
    log(f"ERROR en {contexto}:\n" + traceback.format_exc())


# ---------------------------------------------------------------- ffmpeg
_FFMPEG = None


def ffmpeg_exe():
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    p = shutil.which("ffmpeg")
    if not p:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
    _FFMPEG = p
    return p


def correr(args, cwd=None):
    args = [str(a) for a in args]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("Fallo el comando:\n" + " ".join(args) + "\n" + r.stderr[-3000:])
    return r


def ffmpeg(*args, cwd=None):
    return correr([ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", *args], cwd=cwd)


def duracion_media(path):
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
    if not m:
        raise RuntimeError(f"No pude leer la duracion de {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


# ---------------------------------------------------------------- texto / subtitulos
def fmt_srt(t):
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def partir_lineas(texto, max_chars=42):
    """Parte un texto en lineas de hasta max_chars sin cortar palabras."""
    palabras = texto.split()
    lineas, cur = [], ""
    for w in palabras:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lineas.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lineas.append(cur)
    return lineas


def texto_a_cues(texto, t0, t1, max_chars=42, max_lineas=2):
    """Reparte un texto largo en varios subtitulos (2 lineas max) proporcional a la cantidad de caracteres."""
    lineas = partir_lineas(texto, max_chars)
    grupos = [lineas[i:i + max_lineas] for i in range(0, len(lineas), max_lineas)]
    total = sum(len(" ".join(g)) for g in grupos) or 1
    cues, t = [], t0
    for g in grupos:
        d = (t1 - t0) * len(" ".join(g)) / total
        cues.append((t, t + d, "\n".join(g)))
        t += d
    return cues


def escribir_srt(path, cues):
    out = []
    for i, (a, b, txt) in enumerate(cues, 1):
        out.append(f"{i}\n{fmt_srt(a)} --> {fmt_srt(b)}\n{txt}\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def normalizar_palabras(txt):
    # sirve para cualquier alfabeto (latino, cirilico...): saca acentos y signos, deja letras y numeros
    txt = unicodedata.normalize("NFKD", txt.lower().replace("’", "'"))
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", txt)


def wer(ref, hyp):
    r, h = normalizar_palabras(ref), normalizar_palabras(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def ejecutar_principal(nombre_paso, funcion):
    cfg = cargar_config()
    Log.iniciar(carpeta_proceso(cfg) / "log.txt")
    log("=" * 70)
    log(f"{nombre_paso} | carpeta: {cfg['carpeta_videos']}")
    try:
        funcion(cfg)
    except Exception:
        log_error(nombre_paso)
        sys.exit(1)
