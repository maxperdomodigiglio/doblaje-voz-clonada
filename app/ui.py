"""Interfaz del doblaje: carpeta, idioma y voz; corre los pasos 1 -> 2 -> 3 sin API key,
con progreso por video, pausa y reanudacion."""
import html
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr

from canto import motor_instalado
from comun import APP_DIR, EXT_AUDIO, EXT_MEDIOS, IDIOMAS, archivo_traduccion, slug_de

AJUSTES_UI = APP_DIR / "ui_ajustes.json"
BANDERAS = {"en": "🇬🇧", "de": "🇩🇪", "ru": "🇷🇺", "es": "🇪🇸"}
OPC_IDIOMA = [f"{BANDERAS[k]} {v[3]}" for k, v in IDIOMAS.items()]           # "🇬🇧 Inglés"
COD_DE = {f"{BANDERAS[k]} {v[3]}": k for k, v in IDIOMAS.items()}
COD_DE.update({v[3]: k for k, v in IDIOMAS.items()})                        # tambien "Inglés" a secas
VOZ_VIDEO, VOZ_MUESTRA = "🎬 La voz original de cada video o audio", "🎤 Una muestra de voz propia"
ORIGEN_AUTO = "🔎 Detectar automáticamente"
ORIGENES_UI = {ORIGEN_AUTO: "auto", "Español": "es", "Inglés": "en", "Alemán": "de", "Ruso": "ru",
               "Francés": "fr", "Italiano": "it", "Portugués": "pt"}
UI_DE_ORIGEN = {v: k for k, v in ORIGENES_UI.items()}
EXTS_TXT = ".mp4, .mov, .mkv, .mp3, .wav, .m4a…"
PASOS = {
    1: ("paso1_preparar.py", "Preparar", "separa la voz y transcribe"),
    2: ("paso2_traducir.py", "Traducir", "corrige y traduce con el modelo local"),
    3: ("paso3_doblar.py", "Doblar", "voz clonada, mezcla y video final"),
}
PESO = {1: 0.3, 2: 0.2, 3: 0.5}

_proc = None
_ocupado = threading.Lock()
RUN = {"activo": False, "pausado": False, "detener": False, "pasos": [], "inicio": 0.0, "pausa_acum": 0.0,
       "pausa_desde": None, "prog": {}, "estado": "", "paso": None, "carpeta": "", "cod": "en", "msg": ""}


# ---------------------------------------------------------------- ajustes
def cargar_ajustes():
    base = {}
    try:
        base = json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    aj = {"carpeta": os.path.expandvars(base.get("carpeta_videos", "")), "idioma": OPC_IDIOMA[0], "voz": VOZ_VIDEO,
          "contexto": "", "origen": ORIGEN_AUTO}
    try:
        aj.update(json.loads(AJUSTES_UI.read_text(encoding="utf-8")))
    except Exception:
        pass
    if aj["idioma"] not in OPC_IDIOMA:
        cod = COD_DE.get(aj["idioma"], "en")
        aj["idioma"] = f"{BANDERAS[cod]} {IDIOMAS[cod][3]}"
    if aj["voz"] not in (VOZ_VIDEO, VOZ_MUESTRA):
        aj["voz"] = VOZ_MUESTRA if "muestra" in aj["voz"].lower() else VOZ_VIDEO
    if aj.get("origen") not in ORIGENES_UI:
        aj["origen"] = ORIGEN_AUTO
    # lo propio de cada carpeta (de que trata y el idioma original) se guarda en <carpeta>/_proceso
    pc = ajustes_carpeta(aj["carpeta"])
    aj["contexto"] = pc.get("contexto", "")
    if pc.get("idioma_origen") in UI_DE_ORIGEN:
        aj["origen"] = UI_DE_ORIGEN[pc["idioma_origen"]]
    if "cantar" in pc:
        aj["cantar"] = bool(pc["cantar"])
    return aj


def ajustes_carpeta(carpeta):
    try:
        return json.loads((Path(carpeta or "") / "_proceso" / "ajustes_ui.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def guardar_ajustes(**kw):
    aj = cargar_ajustes()
    aj.update(kw)
    AJUSTES_UI.write_text(json.dumps(aj, ensure_ascii=False, indent=2), encoding="utf-8")


def muestra_lista(carpeta):
    d = Path(carpeta or "") / "_proceso" / "voz_muestra"
    if (d / "ref.wav").exists() and (d / "info.json").exists():
        try:
            return json.loads((d / "info.json").read_text(encoding="utf-8")).get("archivo", "muestra")
        except Exception:
            return "muestra"
    return None


# ---------------------------------------------------------------- estado de la carpeta
def videos_de(carpeta, cod):
    """Lista de videos con lo que ya esta hecho en disco: [{nombre, slug, hecho:{1,2,3: bool}, auto}]."""
    base = Path(carpeta or "")
    if not carpeta or not base.is_dir():
        return None
    proc, sal = base / "_proceso", base / "salida"
    por_video = {}
    if proc.is_dir():
        for d in proc.iterdir():
            try:
                por_video[json.loads((d / "es.json").read_text(encoding="utf-8"))["video"]] = d
            except Exception:
                pass
    suf = IDIOMAS[cod][2]
    out = []
    for v in sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() in EXT_MEDIOS):
        d = por_video.get(v.name, proc / slug_de(v.name))
        t = archivo_traduccion(d, {"idioma_codigo": cod})
        audio = v.suffix.lower() in EXT_AUDIO
        finales = [sal / f"{v.stem} [{suf}].{x}" for x in (("mp3", "m4a") if audio else ("mp4",))]
        auto = None
        if t.exists():
            try:
                data = json.loads(t.read_text(encoding="utf-8"))
                auto = bool(data.get("auto") and not data.get("curado"))
            except Exception:
                pass
        canciones = 0
        if (d / "canto.json").exists():
            try:
                canciones = len(json.loads((d / "canto.json").read_text(encoding="utf-8")).get("tramos", []))
            except Exception:
                pass
        out.append({"nombre": v.stem, "slug": d.name, "audio": audio, "canciones": canciones,
                    "hecho": {1: (d / "es.json").exists(), 2: t.exists(), 3: any(f.exists() for f in finales)},
                    "auto": auto})
    return out


def _fmt_t(seg):
    seg = int(max(0, seg))
    h, m, s = seg // 3600, seg % 3600 // 60, seg % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _avance_video(v, pasos):
    """% de un video para los pasos de esta corrida, y el paso/detalle en curso."""
    tot, peso_t, actual = 0.0, 0.0, None
    for p in pasos:
        pct, det = RUN["prog"].get((v["slug"], p), (None, ""))
        if pct is None:
            pct = 100.0 if (v["hecho"][p] and RUN["estado"] != "corriendo") else 0.0
        tot += PESO[p] * pct
        peso_t += PESO[p]
        if 0 < pct < 100 and actual is None:
            actual = (p, pct, det)
    return (tot / peso_t if peso_t else 0.0), actual


def panel(carpeta, idioma):
    cod = COD_DE.get(idioma, "en")
    vids = videos_de(carpeta, cod)
    if vids is None:
        return ('<div class="vacio">📁 Elegí la carpeta donde están los videos o audios para empezar.</div>')
    if not vids:
        return f'<div class="vacio">No hay videos ni audios en esta carpeta ({EXTS_TXT}).</div>'

    corriendo = RUN["estado"] in ("corriendo", "pausa") and RUN["carpeta"] == carpeta and RUN["cod"] == cod
    pasos = RUN["pasos"] if corriendo or (RUN["estado"] and RUN["carpeta"] == carpeta) else [1, 2, 3]
    partes = []

    # cabecera general
    if RUN["estado"] and RUN["carpeta"] == carpeta and RUN["cod"] == cod:
        avs = [_avance_video(v, pasos)[0] for v in vids]
        total = sum(avs) / len(avs) if avs else 0
        if RUN["estado"] == "listo":
            total = 100.0
        activo = time.time() - RUN["inicio"] - RUN["pausa_acum"] - (
            time.time() - RUN["pausa_desde"] if RUN["pausado"] and RUN["pausa_desde"] else 0)
        eta = ""
        if RUN["estado"] == "corriendo" and 2 <= total < 100:
            eta = f" · faltan ~{_fmt_t(activo * (100 - total) / total)}"
        chip = {"corriendo": ("chip-run", "En curso"), "pausa": ("chip-pausa", "En pausa"),
                "listo": ("chip-ok", "Terminado"), "detenido": ("chip-pausa", "Detenido"),
                "error": ("chip-err", "Con errores")}.get(RUN["estado"], ("chip-run", ""))
        hecho_n = sum(1 for a in avs if a >= 99.9)
        rayas = " rayas" if RUN["estado"] == "corriendo" else ""
        msg = f'<div class="msg">{html.escape(RUN["msg"])}</div>' if RUN["msg"] else ""
        partes.append(
            f'<div class="general"><div class="fila"><span class="chip {chip[0]}">{chip[1]}</span>'
            f'<span class="titulo-gen">{total:.0f}% · {hecho_n} de {len(vids)} archivos</span>'
            f'<span class="tiempo">⏱ {_fmt_t(activo)}{eta}</span></div>'
            f'<div class="barra grande"><div class="relleno{rayas}" style="width:{total:.1f}%"></div></div>{msg}</div>')

    # una tarjeta por video
    for v in vids:
        pct, actual = _avance_video(v, pasos)
        chips = []
        for p in (1, 2, 3):
            pp, _ = RUN["prog"].get((v["slug"], p), (None, ""))
            if pp is None:
                pp = 100.0 if v["hecho"][p] else 0.0
            if pp >= 99.9:
                cls, txt = "ok", f"✓ {PASOS[p][1]}"
            elif pp > 0:
                cls, txt = "run", f"{PASOS[p][1]} {pp:.0f}%"
            else:
                cls, txt = "pend", PASOS[p][1]
            if p == 2 and v["auto"] and pp >= 99.9:
                txt += " · a revisar"
            chips.append(f'<span class="paso {cls}">{txt}</span>')
        det = ""
        if actual and corriendo:
            det = f'<div class="detalle">{"⏸ " if RUN["pausado"] else "⚙️ "}{html.escape(actual[2] or "")}</div>'
        listo = all(v["hecho"].values()) and not actual
        icono = "✅" if listo else ("🔄" if actual else ("🎧" if v.get("audio") else "🎞️"))
        rayas = " rayas" if (actual and corriendo and not RUN["pausado"]) else ""
        partes.append(
            f'<div class="video{" activo" if actual and corriendo else ""}"><div class="fila">'
            f'<span class="nombre" title="{html.escape(v["nombre"])}">{icono} {html.escape(v["nombre"])}</span>'
            + (f'<span class="musica" title="partes cantadas">🎵 {v["canciones"]}</span>' if v.get("canciones") else "")
            + f'<span class="pct">{pct:.0f}%</span></div>'
            f'<div class="barra"><div class="relleno{rayas}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="pasos">{"".join(chips)}</div>{det}</div>')
    return "".join(partes)


# ---------------------------------------------------------------- acciones
def elegir_carpeta(actual):
    codigo = ("import tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk(); r.withdraw()\n"
              "r.attributes('-topmost', True)\n"
              f"p = filedialog.askdirectory(title='Elegí la carpeta con los videos o audios a traducir', initialdir={actual!r})\n"
              "print(p or '')")
    try:
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=900)
        p = r.stdout.strip()
        if p:
            return os.path.normpath(p)
        if r.returncode != 0:
            gr.Warning("No pude abrir el selector de carpetas: pegá la ruta en el cuadro.")
    except Exception:
        gr.Warning("No pude abrir el selector de carpetas: pegá la ruta en el cuadro.")
    return actual


def abrir_salida(carpeta):
    p = Path(carpeta or "") / "salida"
    if not p.is_dir():
        gr.Warning("Todavía no hay nada terminado (la carpeta 'salida' se crea al doblar).")
        return
    try:
        os.startfile(str(p))  # Windows
    except AttributeError:
        subprocess.Popen(["xdg-open", str(p)])


def _suspender(pid, pausar):
    try:
        import psutil
        procs = [psutil.Process(pid)]
        procs += procs[0].children(recursive=True)
        for q in procs:
            try:
                q.suspend() if pausar else q.resume()
            except Exception:
                pass
        return
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x0800, False, pid)  # PROCESS_SUSPEND_RESUME
        (ctypes.windll.ntdll.NtSuspendProcess if pausar else ctypes.windll.ntdll.NtResumeProcess)(h)
        ctypes.windll.kernel32.CloseHandle(h)
    else:
        os.kill(pid, signal.SIGSTOP if pausar else signal.SIGCONT)


def pausar_continuar():
    p = _proc
    if not RUN["activo"] or p is None or p.poll() is not None:
        gr.Info("No hay nada corriendo para pausar.")
        return gr.update(value="⏸ Pausar")
    if not RUN["pausado"]:
        _suspender(p.pid, True)
        RUN.update(pausado=True, pausa_desde=time.time(), estado="pausa")
        gr.Info("En pausa: la placa de video y el procesador quedan libres. La memoria sigue reservada; "
                "si necesitás liberarla toda, usá Detener (después 'Hacer todo' sigue donde quedó).")
        return gr.update(value="▶ Continuar", variant="primary")
    _suspender(p.pid, False)
    RUN["pausa_acum"] += time.time() - (RUN["pausa_desde"] or time.time())
    RUN.update(pausado=False, pausa_desde=None, estado="corriendo")
    return gr.update(value="⏸ Pausar", variant="secondary")


def detener():
    RUN["detener"] = True
    p = _proc
    if p and p.poll() is None:
        if RUN["pausado"]:
            _suspender(p.pid, False)
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            p.terminate()
    return gr.update(value="⏸ Pausar", variant="secondary")


def _correr(script, args, salida):
    """Corre un paso en un proceso aparte (al terminar se libera toda la memoria de la placa)."""
    global _proc
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    _proc = subprocess.Popen([sys.executable, script, *args], cwd=str(APP_DIR), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

    def linea_lista(txt, por_cr):
        if txt.startswith("@@P "):
            try:
                ev = json.loads(txt[4:])
                RUN["prog"][(ev["cap"], ev["paso"])] = (float(ev["pct"]), ev.get("det", ""))
            except Exception:
                pass
            return
        if por_cr and salida and salida[-1].startswith("~"):
            salida[-1] = "~" + txt  # barras de progreso de descargas: piso la linea anterior
        else:
            salida.append(("~" + txt) if por_cr else txt)

    def leer():
        buf = b""
        while True:
            ch = _proc.stdout.read(1)
            if not ch:
                break
            if ch in (b"\n", b"\r"):
                txt = buf.decode("utf-8", "replace").rstrip()
                buf = b""
                if txt:
                    linea_lista(txt, ch == b"\r")
            else:
                buf += ch
        if buf:
            linea_lista(buf.decode("utf-8", "replace").rstrip(), False)

    t = threading.Thread(target=leer, daemon=True)
    t.start()
    return t


def _registro(salida):
    return "\n".join(l.lstrip("~") for l in salida)


def ejecutar(pasos, carpeta, idioma, voz, archivo_voz, contexto, forzar, origen=ORIGEN_AUTO, cantar=False):
    carpeta = (carpeta or "").strip().strip('"')
    if not carpeta or not Path(carpeta).is_dir():
        raise gr.Error("Elegí primero la carpeta donde están los videos o audios.")
    if not any(p.suffix.lower() in EXT_MEDIOS for p in Path(carpeta).iterdir() if p.is_file()):
        raise gr.Error(f"En esa carpeta no hay videos ni audios ({EXTS_TXT}).")
    cod = COD_DE.get(idioma, "en")
    lista = list(pasos)
    if voz == VOZ_MUESTRA and 3 in lista and not archivo_voz and not muestra_lista(carpeta):
        raise gr.Error("Elegiste usar una muestra de voz propia: subí un audio o video con la voz (10-30 s).")
    if not _ocupado.acquire(blocking=False):
        raise gr.Error("Ya hay un proceso corriendo. Esperá a que termine, o tocá Detener.")
    salida = deque(maxlen=500)
    try:
        guardar_ajustes(carpeta=carpeta, idioma=idioma, voz=voz, origen=origen, cantar=bool(cantar))
        proc = Path(carpeta) / "_proceso"
        proc.mkdir(parents=True, exist_ok=True)
        ajustes = {"carpeta_videos": carpeta, "idioma_codigo": cod,
                   "idioma_origen": ORIGENES_UI.get(origen, "auto"), "cantar": bool(cantar and motor_instalado())}
        if (contexto or "").strip():
            ajustes["contexto"] = contexto.strip()
        if voz == VOZ_MUESTRA and 3 in lista:
            if archivo_voz:
                lista.insert(lista.index(3), "voz")
            ajustes["voz_muestra"] = str(proc / "voz_muestra")
        aj_path = proc / "ajustes_ui.json"
        aj_path.write_text(json.dumps(ajustes, ensure_ascii=False, indent=2), encoding="utf-8")

        RUN.update(activo=True, pausado=False, detener=False, pasos=[p for p in lista if p != "voz"],
                   inicio=time.time(), pausa_acum=0.0, pausa_desde=None, prog={}, estado="corriendo",
                   carpeta=carpeta, cod=cod, msg="")
        for paso in lista:
            if RUN["detener"]:
                break
            if paso == "voz":
                script, titulo, extra = "voz_muestra.py", "Muestra de voz", [f"--archivo={archivo_voz}"]
                RUN["msg"] = "Preparando la muestra de voz…"
            else:
                script, titulo, _ = PASOS[paso]
                extra = ["--forzar"] if (forzar and paso == 2) else []
                RUN["msg"] = f"Paso {paso} de 3 · {PASOS[paso][1]}: {PASOS[paso][2]}"
            if os.environ.get("DOBLAJE_SIMULAR"):
                extra.append("--simular")  # solo para pruebas
            RUN["paso"] = paso
            salida.append("")
            salida.append(f"=== {RUN['msg']} ===")
            hilo = _correr(script, [f"--config={aj_path}", *extra], salida)
            while hilo.is_alive() or _proc.poll() is None:
                yield panel(carpeta, idioma), _registro(salida), gr.update()
                time.sleep(1.0)
            hilo.join(timeout=2)
            codigo = _proc.returncode
            if RUN["detener"]:
                RUN.update(estado="detenido", msg="Detenido. Tocá 'Hacer todo' para seguir desde donde quedó.")
                salida.append("=== Detenido ===")
                break
            if codigo != 0:
                RUN.update(estado="error", msg=f"El paso «{titulo}» terminó con errores. El detalle está abajo "
                                                f"y en _proceso/log.txt.")
                salida.append(f"=== {titulo}: terminó con errores (código {codigo}) ===")
                break
        else:
            RUN.update(estado="listo", msg="¡Listo! Los resultados están en la carpeta 'salida'." if 3 in lista
                       else "Listo.")
            salida.append("=== Listo ===")
        RUN.update(activo=False, pausado=False)
        yield panel(carpeta, idioma), _registro(salida), gr.update(value="⏸ Pausar", variant="secondary")
    finally:
        RUN["activo"] = False
        _ocupado.release()


# ---------------------------------------------------------------- interfaz
CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#hero {padding: 26px 30px; border-radius: 20px; margin-bottom: 6px; color: white;
       background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
       box-shadow: 0 10px 30px rgba(79,70,229,.25);}
#hero h1 {margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -.5px; color: white;}
#hero p {margin: 6px 0 0; opacity: .92; font-size: 15px; color: white;}
.tarjeta {border-radius: 18px !important; padding: 18px !important; box-shadow: 0 4px 18px rgba(15,23,42,.06);
          border: 1px solid rgba(100,116,139,.15) !important;}
.seccion {font-weight: 700; font-size: 15px; margin: 2px 0 -4px; opacity: .85;}
#btn-todo {font-size: 18px !important; padding: 14px !important; border-radius: 14px !important;
           background: linear-gradient(90deg, #4f46e5, #7c3aed) !important; color: white !important; border: none !important;}
.vacio {padding: 40px 20px; text-align: center; opacity: .7; font-size: 15px;}
.general {padding: 16px 18px; border-radius: 16px; margin-bottom: 14px;
          background: linear-gradient(135deg, rgba(79,70,229,.08), rgba(219,39,119,.06));
          border: 1px solid rgba(79,70,229,.18);}
.general .msg {margin-top: 8px; font-size: 13px; opacity: .8;}
.fila {display: flex; align-items: center; gap: 10px;}
.titulo-gen {font-weight: 700; font-size: 16px;}
.tiempo {margin-left: auto; font-size: 13px; opacity: .75; white-space: nowrap;}
.chip {font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 999px; color: white;}
.chip-run {background: #4f46e5;} .chip-pausa {background: #f59e0b;} .chip-ok {background: #16a34a;}
.chip-err {background: #dc2626;}
.barra {height: 9px; border-radius: 999px; background: rgba(100,116,139,.18); overflow: hidden; margin-top: 8px;}
.barra.grande {height: 14px;}
.relleno {height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #a855f7, #22c55e);
          transition: width .8s ease;}
.relleno.rayas {background-image: linear-gradient(45deg, rgba(255,255,255,.25) 25%, transparent 25%,
          transparent 50%, rgba(255,255,255,.25) 50%, rgba(255,255,255,.25) 75%, transparent 75%),
          linear-gradient(90deg, #6366f1, #a855f7); background-size: 22px 22px, 100% 100%;
          animation: correr 1s linear infinite;}
@keyframes correr {from {background-position: 0 0, 0 0;} to {background-position: 22px 0, 0 0;}}
.video {padding: 12px 14px; border-radius: 14px; margin-bottom: 10px; border: 1px solid rgba(100,116,139,.18);}
.video.activo {border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,.15);}
.nombre {font-weight: 600; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;}
.pct {font-weight: 700; font-size: 14px; color: #6366f1;}
.musica {font-size: 12px; font-weight: 600; color: #a855f7; background: #f3e8ff; border-radius: 999px; padding: 1px 8px; margin-left: auto; margin-right: 8px;}
.pasos {display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px;}
.paso {font-size: 12px; padding: 2px 9px; border-radius: 999px; border: 1px solid rgba(100,116,139,.25);}
.paso.ok {background: rgba(22,163,74,.12); color: #15803d; border-color: rgba(22,163,74,.3);}
.paso.run {background: rgba(99,102,241,.12); color: #4f46e5; border-color: rgba(99,102,241,.35); font-weight: 600;}
.paso.pend {opacity: .55;}
.detalle {margin-top: 6px; font-size: 13px; opacity: .8;}
#panel {max-height: 640px; overflow-y: auto; padding-right: 4px;}
footer {display: none !important;}
#btn-stop {color: #dc2626 !important;}
#pasos-sueltos {flex-wrap: nowrap !important;}
#pasos-sueltos button {opacity: .85;}
"""


def armar():
    aj = cargar_ajustes()
    with gr.Blocks(title="Doblaje con voz clonada") as demo:
        gr.HTML('<div id="hero"><h1>🎙️ Doblaje con voz clonada</h1>'
                '<p>Traducí y doblá tus videos o audios al inglés, alemán, ruso o castellano, con subtítulos. '
                'Todo corre en tu compu, sin API key.</p></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="tarjeta"):
                gr.HTML('<div class="seccion">1 · Carpeta con los videos o audios</div>')
                with gr.Row():
                    carpeta = gr.Textbox(value=aj["carpeta"], show_label=False, scale=5,
                                         placeholder="C:\\Users\\...\\Desktop\\mis videos")
                    b_elegir = gr.Button("📁 Elegir", scale=1, min_width=110)
                ultima = gr.State(aj["carpeta"])
                gr.HTML('<div class="seccion">2 · Idioma del doblaje</div>')
                idioma = gr.Radio(OPC_IDIOMA, value=aj["idioma"], show_label=False)
                gr.HTML('<div class="seccion">3 · Voz</div>')
                voz = gr.Radio([VOZ_VIDEO, VOZ_MUESTRA], value=aj["voz"], show_label=False)
                archivo_voz = gr.File(label="Muestra de voz: 10-30 s de una sola persona hablando, sin música",
                                      file_types=["audio", "video"], type="filepath",
                                      visible=aj["voz"] == VOZ_MUESTRA, height=110)
                info_voz = gr.Markdown(visible=aj["voz"] == VOZ_MUESTRA)
                hay_canto = motor_instalado()
                cantar = gr.Checkbox(
                    label="🎵 Cantar las canciones: detecta las partes cantadas y las canta traducidas, "
                          "con la misma melodía",
                    value=bool(aj.get("cantar", True)) and hay_canto, interactive=hay_canto,
                    info=None if hay_canto else "Para usarlo, en Pinokio tocá '🎵 Instalar canto' (una sola vez).")
                with gr.Accordion("⚙️ Opciones avanzadas", open=False):
                    origen = gr.Dropdown(list(ORIGENES_UI), value=aj["origen"],
                                         label="Idioma en que hablan los videos/audios originales")
                    contexto = gr.Textbox(label="De qué trata (ayuda a traducir mejor: quién habla y de qué)",
                                          value=aj.get("contexto", ""), lines=2,
                                          placeholder="Ej.: Habla un psicólogo, en primera persona, sobre arte y "
                                                      "salud mental. Se guarda para esta carpeta.")
                    forzar = gr.Checkbox(label="Rehacer las traducciones automáticas (las revisadas no se tocan)",
                                         value=False)
                b_todo = gr.Button("▶  Hacer todo", elem_id="btn-todo")
                with gr.Row(elem_id="pasos-sueltos"):
                    b1 = gr.Button("1 · Preparar", size="sm", min_width=60)
                    b2 = gr.Button("2 · Traducir", size="sm", min_width=60)
                    b3 = gr.Button("3 · Doblar", size="sm", min_width=60)
                with gr.Row():
                    b_pausa = gr.Button("⏸ Pausar", variant="secondary")
                    b_stop = gr.Button("⏹ Detener", variant="stop", elem_id="btn-stop")
                b_salida = gr.Button("📂 Abrir los resultados", variant="secondary")
            with gr.Column(scale=7, elem_classes="tarjeta"):
                with gr.Row():
                    gr.HTML('<div class="seccion">Progreso</div>')
                    b_ref = gr.Button("🔄", size="sm", min_width=50, scale=0)
                tablero = gr.HTML(panel(aj["carpeta"], aj["idioma"]), elem_id="panel")
                with gr.Accordion("🧾 Detalle técnico", open=False):
                    registro = gr.Textbox(show_label=False, lines=16, max_lines=16, autoscroll=True,
                                          interactive=False)

        def ver_voz(v, c):
            m = muestra_lista(c)
            txt = (f"✔️ Ya hay una muestra preparada en esta carpeta: **{m}**. Subí otra para cambiarla."
                   if m else "Todavía no hay una muestra en esta carpeta.")
            return gr.update(visible=v == VOZ_MUESTRA), gr.update(visible=v == VOZ_MUESTRA, value=txt)

        def cambio_carpeta(c, ult):
            # al pasar a otra carpeta se cargan su tema, su idioma original y si se cantan las canciones
            if (c or "") == (ult or ""):
                return gr.update(), gr.update(), gr.update(), ult
            pc = ajustes_carpeta(c)
            org = UI_DE_ORIGEN.get(pc.get("idioma_origen"), ORIGEN_AUTO)
            can = gr.update(value=bool(pc["cantar"])) if ("cantar" in pc and hay_canto) else gr.update()
            return gr.update(value=pc.get("contexto", "")), gr.update(value=org), can, c

        voz.change(ver_voz, [voz, carpeta], [archivo_voz, info_voz])
        b_elegir.click(elegir_carpeta, carpeta, carpeta).then(panel, [carpeta, idioma], tablero) \
            .then(ver_voz, [voz, carpeta], [archivo_voz, info_voz]) \
            .then(cambio_carpeta, [carpeta, ultima], [contexto, origen, cantar, ultima])
        carpeta.submit(panel, [carpeta, idioma], tablero).then(cambio_carpeta, [carpeta, ultima],
                                                               [contexto, origen, cantar, ultima])
        carpeta.blur(panel, [carpeta, idioma], tablero).then(cambio_carpeta, [carpeta, ultima],
                                                             [contexto, origen, cantar, ultima])
        idioma.change(panel, [carpeta, idioma], tablero)
        b_ref.click(panel, [carpeta, idioma], tablero)

        def hacer(pasos):
            def correr_pasos(carpeta_, idioma_, voz_, archivo_, contexto_, forzar_, origen_, cantar_):
                yield from ejecutar(pasos, carpeta_, idioma_, voz_, archivo_, contexto_, forzar_, origen_, cantar_)
            return correr_pasos

        entradas = [carpeta, idioma, voz, archivo_voz, contexto, forzar, origen, cantar]
        for boton, pasos in ((b_todo, (1, 2, 3)), (b1, (1,)), (b2, (2,)), (b3, (3,))):
            boton.click(hacer(pasos), entradas, [tablero, registro, b_pausa])
        b_pausa.click(pausar_continuar, None, b_pausa, concurrency_limit=None, queue=False)
        b_stop.click(detener, None, b_pausa, concurrency_limit=None, queue=False)
        b_salida.click(abrir_salida, carpeta, None, queue=False)
    return demo


if __name__ == "__main__":
    tema = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate", radius_size="lg")
    armar().queue(default_concurrency_limit=2).launch(server_name="127.0.0.1", inbrowser=False, theme=tema, css=CSS)
