"""Prepara una muestra de voz subida en la interfaz para clonarla en el Paso 3.

  python voz_muestra.py --config=<ajustes.json> --archivo=<audio o video>

Deja en <carpeta>/_proceso/voz_muestra/:  ref.wav (8-14 s de habla limpia) y ref.txt (lo que dice).
"""
import hashlib
import shutil
import sys
from pathlib import Path

from comun import carpeta_proceso, ejecutar_principal, escribir_json, ffmpeg, leer_json, log, progreso

ARCHIVO = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--archivo=")), None)


def preparar_muestra(cfg, archivo):
    from paso1_preparar import armar_bloques, elegir_referencia, separar, transcribir

    origen = Path(archivo)
    if not origen.exists():
        raise FileNotFoundError(f"No encuentro la muestra de voz: {origen}")
    d = carpeta_proceso(cfg) / "voz_muestra"
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(origen.read_bytes()).hexdigest()[:12]
    info_p = d / "info.json"
    if info_p.exists() and leer_json(info_p).get("hash") == h and (d / "ref.wav").exists():
        log(f"Muestra de voz ya preparada ({leer_json(info_p).get('archivo')}), la reutilizo")
        return d
    copia = d / ("original" + origen.suffix.lower())
    if origen.resolve() != copia.resolve():
        shutil.copyfile(origen, copia)
    log(f"Preparando la muestra de voz: {origen.name}")
    progreso("voz", "voz", 5, "preparando la muestra de voz")
    wav = d / "muestra.wav"
    ffmpeg("-i", copia, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", wav)
    voz = d / "voz.flac"
    if cfg.get("limpiar_muestra", True):
        separar(wav, voz, d / "fondo.flac", cfg)  # saca musica y ruido de fondo
    else:
        ffmpeg("-i", wav, "-ac", "1", voz)
    progreso("voz", "voz", 50, "transcribiendo la muestra")
    cfg_t = dict(cfg, idioma_origen=None, whisper_prompt="")  # la muestra puede estar en cualquier idioma
    segmentos = transcribir(voz, cfg_t)
    if not segmentos:
        raise RuntimeError("No se escucha habla en la muestra de voz")
    ref = elegir_referencia(armar_bloques(segmentos, 14.0), segmentos, voz, d, cfg)
    escribir_json(info_p, {"hash": h, "archivo": origen.name, "ref": ref})
    for f in (wav, d / "fondo.flac"):
        f.unlink(missing_ok=True)
    dur = ref["end"] - ref["start"]
    progreso("voz", "voz", 100, "muestra lista")
    log(f"Muestra lista: {dur:.1f}s de voz -> \"{ref['texto'][:80]}\"")
    if dur < 6:
        log("  OJO: la muestra es corta (ideal 10-15 s de una sola persona hablando, sin musica)")
    return d


def main(cfg):
    if not ARCHIVO:
        raise SystemExit("Falta --archivo=")
    preparar_muestra(cfg, ARCHIVO)


if __name__ == "__main__":
    ejecutar_principal("MUESTRA DE VOZ", main)
