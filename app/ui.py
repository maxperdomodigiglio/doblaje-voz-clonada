"""Interfaz del Doblaje Gregorio: elegir carpeta, idioma y voz, y correr los pasos 1 -> 2 -> 3 sin API key."""
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr

from comun import APP_DIR, EXT_VIDEO, IDIOMAS, slug_de

AJUSTES_UI = APP_DIR / "ui_ajustes.json"
NOMBRE_A_COD = {v[3]: k for k, v in IDIOMAS.items()}  # "Inglés" -> "en"
VOZ_VIDEO, VOZ_MUESTRA = "Voz original de cada video", "Muestra de voz propia"
PASOS = {
    1: ("paso1_preparar.py", "Paso 1 · Preparar (separar voz y transcribir)"),
    2: ("paso2_traducir.py", "Paso 2 · Corregir y traducir (modelo local)"),
    3: ("paso3_doblar.py", "Paso 3 · Doblar y armar los videos"),
}

_proc = None
_detener = threading.Event()
_ocupado = threading.Lock()


# ---------------------------------------------------------------- ajustes
def cargar_ajustes():
    base = {}
    try:
        base = json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    aj = {"carpeta": base.get("carpeta_videos", ""), "idioma": "Inglés", "voz": VOZ_VIDEO, "contexto": ""}
    try:
        aj.update(json.loads(AJUSTES_UI.read_text(encoding="utf-8")))
    except Exception:
        pass
    return aj


def guardar_ajustes(**kw):
    aj = cargar_ajustes()
    aj.update(kw)
    AJUSTES_UI.write_text(json.dumps(aj, ensure_ascii=False, indent=2), encoding="utf-8")


def muestra_lista(carpeta):
    d = Path(carpeta) / "_proceso" / "voz_muestra"
    if (d / "ref.wav").exists() and (d / "info.json").exists():
        try:
            return json.loads((d / "info.json").read_text(encoding="utf-8")).get("archivo", "muestra")
        except Exception:
            return "muestra"
    return None


# ---------------------------------------------------------------- estado de la carpeta
def tabla_estado(carpeta, idioma):
    cod = NOMBRE_A_COD.get(idioma, "en")
    suf = IDIOMAS[cod][2]
    base = Path(carpeta or "")
    if not carpeta or not base.is_dir():
        return [["—", "Elegí una carpeta con videos", "", "", ""]]
    videos = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() in EXT_VIDEO)
    if not videos:
        return [["—", "No hay videos en esta carpeta", "", "", ""]]
    proc, sal = base / "_proceso", base / "salida"
    por_video = {}
    if proc.is_dir():
        for d in proc.iterdir():
            try:
                por_video[json.loads((d / "es.json").read_text(encoding="utf-8"))["video"]] = d
            except Exception:
                pass
    filas = []
    for v in videos:
        d = por_video.get(v.name, proc / slug_de(v.name))
        prep = "✓" if (d / "es.json").exists() else "—"
        trad = "—"
        t = d / f"{cod}.json"
        if t.exists():
            try:
                data = json.loads(t.read_text(encoding="utf-8"))
                trad = "automática (a revisar)" if data.get("auto") and not data.get("curado") else "revisada ✓"
            except Exception:
                trad = "?"
        final = "✓" if (sal / f"{v.stem} [{suf}].mp4").exists() else "—"
        filas.append([d.name, v.name, prep, trad, final])
    return filas


# ---------------------------------------------------------------- acciones
def elegir_carpeta(actual):
    codigo = ("import tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk(); r.withdraw()\n"
              "r.attributes('-topmost', True)\n"
              f"p = filedialog.askdirectory(title='Elegí la carpeta con los videos a traducir', initialdir={actual!r})\n"
              "print(p or '')")
    try:
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=900)
        p = r.stdout.strip()
        if p:
            return os.path.normpath(p)
        if r.returncode != 0:
            gr.Warning("No pude abrir el selector de carpetas: pegá la ruta a mano en el cuadro.")
    except Exception:
        gr.Warning("No pude abrir el selector de carpetas: pegá la ruta a mano en el cuadro.")
    return actual


def abrir_salida(carpeta):
    p = Path(carpeta or "") / "salida"
    if not p.is_dir():
        gr.Warning("Todavía no hay carpeta 'salida' (se crea en el Paso 3).")
        return
    try:
        os.startfile(str(p))  # Windows
    except AttributeError:
        subprocess.Popen(["xdg-open", str(p)])


def detener():
    _detener.set()
    p = _proc
    if p and p.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            p.terminate()
    return "Deteniendo..."


def _correr(script, args, salida):
    """Corre un paso en un proceso aparte (asi la memoria de la GPU se libera al terminar)."""
    global _proc
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    _proc = subprocess.Popen([sys.executable, script, *args], cwd=str(APP_DIR), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

    def leer():
        buf = b""
        while True:
            ch = _proc.stdout.read(1)
            if not ch:
                break
            if ch in (b"\n", b"\r"):
                linea = buf.decode("utf-8", "replace").rstrip()
                buf = b""
                if linea:
                    if ch == b"\r" and salida and salida[-1].startswith("~"):
                        salida[-1] = "~" + linea  # barras de progreso: piso la linea anterior
                    else:
                        salida.append(("~" + linea) if ch == b"\r" else linea)
            else:
                buf += ch
        if buf:
            salida.append(buf.decode("utf-8", "replace"))

    t = threading.Thread(target=leer, daemon=True)
    t.start()
    return t


def ejecutar(pasos, carpeta, idioma, voz, archivo_voz, contexto, forzar):
    carpeta = (carpeta or "").strip().strip('"')
    if not carpeta or not Path(carpeta).is_dir():
        raise gr.Error("Elegí primero la carpeta donde están los videos.")
    if not any(p.suffix.lower() in EXT_VIDEO for p in Path(carpeta).iterdir() if p.is_file()):
        raise gr.Error("En esa carpeta no hay videos (.mp4, .mov, .mkv...).")
    if not _ocupado.acquire(blocking=False):
        raise gr.Error("Ya hay un proceso corriendo. Esperá a que termine o tocá Detener.")
    _detener.clear()
    try:
        cod = NOMBRE_A_COD.get(idioma, "en")
        guardar_ajustes(carpeta=carpeta, idioma=idioma, voz=voz, contexto=contexto or "")
        proc = Path(carpeta) / "_proceso"
        proc.mkdir(parents=True, exist_ok=True)
        ajustes = {"carpeta_videos": carpeta, "idioma_codigo": cod}
        if (contexto or "").strip():
            ajustes["contexto"] = contexto.strip()
        lista = list(pasos)
        if voz == VOZ_MUESTRA and 3 in lista:
            if archivo_voz:
                lista.insert(lista.index(3), "voz")
            elif not muestra_lista(carpeta):
                raise gr.Error("Elegiste 'Muestra de voz propia': subí un audio o video con la voz (10-30 s).")
            ajustes["voz_muestra"] = str(proc / "voz_muestra")
        aj_path = proc / "ajustes_ui.json"
        aj_path.write_text(json.dumps(ajustes, ensure_ascii=False, indent=2), encoding="utf-8")

        salida = deque(maxlen=400)
        for paso in lista:
            if _detener.is_set():
                break
            if paso == "voz":
                script, titulo, extra = "voz_muestra.py", "Preparando la muestra de voz", [f"--archivo={archivo_voz}"]
            else:
                script, titulo = PASOS[paso]
                extra = ["--forzar"] if (forzar and paso == 2) else []
            if os.environ.get("DOBLAJE_SIMULAR"):
                extra.append("--simular")  # solo para pruebas
            salida.append("")
            salida.append(f"=== {titulo} ===")
            hilo = _correr(script, [f"--config={aj_path}", *extra], salida)
            while hilo.is_alive() or _proc.poll() is None:
                yield "\n".join(l.lstrip("~") for l in salida), tabla_estado(carpeta, idioma)
                time.sleep(1.0)
            hilo.join(timeout=2)
            codigo = _proc.returncode
            if _detener.is_set():
                salida.append("=== Detenido ===")
                break
            if codigo != 0:
                salida.append(f"=== {titulo}: terminó con errores (código {codigo}). Detalle arriba y en "
                              f"_proceso/log.txt ===")
                break
        else:
            salida.append("")
            salida.append("=== Listo ===" + (" Los videos están en la carpeta 'salida'." if 3 in lista else ""))
        yield "\n".join(l.lstrip("~") for l in salida), tabla_estado(carpeta, idioma)
    finally:
        _ocupado.release()


# ---------------------------------------------------------------- interfaz
def armar():
    aj = cargar_ajustes()
    with gr.Blocks(title="Doblaje Gregorio") as demo:
        gr.Markdown("## Doblaje automático\nTraduce y dobla los videos de una carpeta con voz clonada y subtítulos "
                    "en castellano. Todo corre en tu compu, sin API key. Después lo podemos curar con Claude.")
        with gr.Row():
            carpeta = gr.Textbox(label="Carpeta con los videos", value=aj["carpeta"], scale=5)
            b_elegir = gr.Button("📁 Elegir…", scale=1, min_width=120)
        with gr.Row():
            idioma = gr.Radio(list(NOMBRE_A_COD), value=aj["idioma"], label="Traducir y doblar al")
            voz = gr.Radio([VOZ_VIDEO, VOZ_MUESTRA], value=aj["voz"], label="Voz")
        archivo_voz = gr.File(label="Muestra de voz (audio o video, 10-30 s de una sola persona, sin música)",
                              file_types=["audio", "video"], type="filepath", visible=aj["voz"] == VOZ_MUESTRA)
        info_voz = gr.Markdown(visible=aj["voz"] == VOZ_MUESTRA)
        with gr.Accordion("Opciones", open=False):
            contexto = gr.Textbox(label="Sobre el video (ayuda a traducir mejor: quién habla y de qué)",
                                  value=aj.get("contexto", ""), lines=2,
                                  placeholder="Si lo dejás vacío: el Dr. Gregorio Dunayevich, psicólogo, sobre arte, "
                                              "rock, genios, drogas y suicidio.")
            forzar = gr.Checkbox(label="Rehacer las traducciones automáticas (las revisadas no se tocan)", value=False)
        with gr.Row():
            b_todo = gr.Button("▶ Hacer todo (1 → 2 → 3)", variant="primary", scale=3)
            b1 = gr.Button("1 · Preparar")
            b2 = gr.Button("2 · Traducir")
            b3 = gr.Button("3 · Doblar")
        with gr.Row():
            b_stop = gr.Button("⏹ Detener", variant="stop")
            b_salida = gr.Button("📂 Abrir carpeta de salida")
            b_ref = gr.Button("🔄 Actualizar estado")
        estado = gr.Dataframe(headers=["Capítulo", "Video", "Preparado", "Traducción", "Video final"],
                              value=tabla_estado(aj["carpeta"], aj["idioma"]), interactive=False, wrap=True)
        registro = gr.Textbox(label="Progreso", lines=18, max_lines=18, autoscroll=True, interactive=False)

        def ver_voz(v, c):
            m = muestra_lista(c)
            txt = f"Muestra ya preparada en esta carpeta: **{m}** (subí otra para cambiarla)." if m else \
                "Todavía no hay muestra en esta carpeta."
            return gr.update(visible=v == VOZ_MUESTRA), gr.update(visible=v == VOZ_MUESTRA, value=txt)

        voz.change(ver_voz, [voz, carpeta], [archivo_voz, info_voz])
        b_elegir.click(elegir_carpeta, carpeta, carpeta).then(tabla_estado, [carpeta, idioma], estado) \
            .then(ver_voz, [voz, carpeta], [archivo_voz, info_voz])
        carpeta.submit(tabla_estado, [carpeta, idioma], estado)
        idioma.change(tabla_estado, [carpeta, idioma], estado)
        b_ref.click(tabla_estado, [carpeta, idioma], estado)
        entradas = [carpeta, idioma, voz, archivo_voz, contexto, forzar]
        def hacer(pasos):
            def correr_pasos(carpeta_, idioma_, voz_, archivo_, contexto_, forzar_):
                yield from ejecutar(pasos, carpeta_, idioma_, voz_, archivo_, contexto_, forzar_)
            return correr_pasos

        for boton, pasos in ((b_todo, (1, 2, 3)), (b1, (1,)), (b2, (2,)), (b3, (3,))):
            boton.click(hacer(pasos), entradas, [registro, estado])
        b_stop.click(detener, None, registro)
        b_salida.click(abrir_salida, carpeta, None)
    return demo


if __name__ == "__main__":
    armar().queue(default_concurrency_limit=2).launch(server_name="127.0.0.1", inbrowser=False,
                                                      theme=gr.themes.Soft())
