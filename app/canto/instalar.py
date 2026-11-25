"""Baja el codigo de YingMusic-Singer-Plus (una version fija) a app/canto/YingMusic-Singer-Plus.

Los pesos del modelo (~3 GB) se bajan solos la primera vez que se canta (Hugging Face).
"""
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "ASLP-lab/YingMusic-Singer-Plus"
COMMIT = "baa409c2e7e5e775f09b4e92a220808f4827d2cc"
DESTINO = Path(__file__).resolve().parent / "YingMusic-Singer-Plus"
MARCA = DESTINO / ".version"


def bajar_zip():
    url = f"https://github.com/{REPO}/archive/{COMMIT}.zip"
    print(f"Bajando {url} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf, leido = io.BytesIO(), 0
        while True:
            parte = r.read(1 << 20)
            if not parte:
                break
            buf.write(parte)
            leido += len(parte)
            if total:
                print(f"  {leido / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    tmp = DESTINO.with_name(DESTINO.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(buf) as z:
        z.extractall(tmp)
    raiz = next(p for p in tmp.iterdir() if p.is_dir())
    shutil.rmtree(DESTINO, ignore_errors=True)
    raiz.replace(DESTINO)
    shutil.rmtree(tmp, ignore_errors=True)


def bajar_git():
    shutil.rmtree(DESTINO, ignore_errors=True)
    subprocess.run(["git", "clone", f"https://github.com/{REPO}", str(DESTINO)], check=True)
    subprocess.run(["git", "-C", str(DESTINO), "checkout", COMMIT], check=True)


def main():
    if MARCA.exists() and MARCA.read_text().strip() == COMMIT and (DESTINO / "src" / "YingMusicSinger").is_dir():
        print("YingMusic-Singer-Plus ya estaba bajado.")
        return
    try:
        bajar_zip()
    except Exception as e:
        print(f"No se pudo bajar el zip ({e}); pruebo con git...", flush=True)
        bajar_git()
    MARCA.write_text(COMMIT)
    print(f"Listo: {DESTINO}")


if __name__ == "__main__":
    sys.exit(main())
