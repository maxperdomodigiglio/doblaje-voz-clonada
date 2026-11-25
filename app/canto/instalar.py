"""Baja el codigo de los motores de canto (versiones fijas) a app/canto/:

- YingMusic-Singer-Plus (ASLP-lab): canta una letra nueva sobre la melodia original.
- YingMusic-SVC (GiantAILab, MIT): lleva una voz cantada a otro timbre sin tocar sus notas.

Los pesos de los modelos se bajan de Hugging Face: los del conversor al instalar (canto_motor.py
--bajar-conversor) y los del cantante la primera vez que se canta.
"""
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MOTORES = [
    # (repo de GitHub, commit fijo, carpeta, archivo que tiene que estar)
    ("ASLP-lab/YingMusic-Singer-Plus", "baa409c2e7e5e775f09b4e92a220808f4827d2cc", "YingMusic-Singer-Plus",
     "src/YingMusicSinger"),
    ("GiantAILab/YingMusic-SVC", "4974a80c6044c4557059548409379f6365129f88", "YingMusic-SVC",
     "modules/rmvpe.py"),
]


def bajar_zip(repo, commit, destino):
    url = f"https://github.com/{repo}/archive/{commit}.zip"
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
    tmp = destino.with_name(destino.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(buf) as z:
        z.extractall(tmp)
    raiz = next(p for p in tmp.iterdir() if p.is_dir())
    shutil.rmtree(destino, ignore_errors=True)
    raiz.replace(destino)
    shutil.rmtree(tmp, ignore_errors=True)


def bajar_git(repo, commit, destino):
    shutil.rmtree(destino, ignore_errors=True)
    subprocess.run(["git", "clone", f"https://github.com/{repo}", str(destino)], check=True)
    subprocess.run(["git", "-C", str(destino), "checkout", commit], check=True)


def main():
    for repo, commit, carpeta, prueba in MOTORES:
        destino = AQUI / carpeta
        marca = destino / ".version"
        if marca.exists() and marca.read_text().strip() == commit and (destino / prueba).exists():
            print(f"{carpeta} ya estaba bajado.")
            continue
        try:
            bajar_zip(repo, commit, destino)
        except Exception as e:
            print(f"No se pudo bajar el zip ({e}); pruebo con git...", flush=True)
            bajar_git(repo, commit, destino)
        marca.write_text(commit)
        print(f"Listo: {destino}")


if __name__ == "__main__":
    sys.exit(main())
