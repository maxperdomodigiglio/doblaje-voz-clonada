"""Interfaz del doblaje: carpeta, idioma y voz; corre los pasos 1 -> 2 -> 3 sin API key,
con progreso por video, pausa y reanudacion."""
import base64
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
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
VOZ_VIDEO, VOZ_MUESTRA = "La voz original de cada video o audio", "Una muestra de voz propia"
ORIGEN_AUTO = "Detectar automáticamente"
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
TRADUCTORES_UI = {"Qwen3-4B: rápido, el de siempre": "qwen3-4b",
                  "Qwen3.5-9B: más inteligente (corrige, traduce y adapta letras)": "qwen3.5-9b",
                  "Hy-MT2-7B: especialista en traducir": "hy-mt2-7b"}
TRADUCTOR_BASE = "Qwen3-4B: rápido, el de siempre"
IDIOMAS_TEXTO = {"Detectar solo": "auto", "Español": "es", "Inglés": "en", "Alemán": "de", "Ruso": "ru",
                 "Francés": "fr", "Italiano": "it", "Portugués": "pt"}

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
          "contexto": "", "origen": ORIGEN_AUTO, "texto_idioma": "Detectar solo", "traductor": TRADUCTOR_BASE}
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
    if aj.get("texto_idioma") not in IDIOMAS_TEXTO:
        aj["texto_idioma"] = "Detectar solo"
    if aj.get("traductor") not in TRADUCTORES_UI:
        aj["traductor"] = TRADUCTOR_BASE
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
            pct = 100.0 if v["hecho"][p] else 0.0
        tot += PESO[p] * pct
        peso_t += PESO[p]
        if 0 < pct < 100 and actual is None:
            actual = (p, pct, det)
    return (tot / peso_t if peso_t else 0.0), actual


def panel(carpeta, idioma):
    cod = COD_DE.get(idioma, "en")
    vids = videos_de(carpeta, cod)
    if vids is None:
        return '<div class="vacio">Elegí la carpeta donde están los videos o audios para empezar.</div>'
    if not vids:
        return f'<div class="vacio">No hay videos ni audios en esta carpeta ({EXTS_TXT}).</div>'

    corriendo = RUN["estado"] in ("corriendo", "pausa") and RUN["carpeta"] == carpeta and RUN["cod"] == cod
    pasos = RUN["pasos"] if corriendo or (RUN["estado"] and RUN["carpeta"] == carpeta) else [1, 2, 3]
    partes = []

    # resumen de la corrida
    if RUN["estado"] and RUN["carpeta"] == carpeta and RUN["cod"] == cod:
        avs = [_avance_video(v, pasos)[0] for v in vids]
        total = sum(avs) / len(avs) if avs else 0
        if RUN["estado"] == "listo":
            total = 100.0
        activo = time.time() - RUN["inicio"] - RUN["pausa_acum"] - (
            time.time() - RUN["pausa_desde"] if RUN["pausado"] and RUN["pausa_desde"] else 0)
        tiempo = _fmt_t(activo)
        if RUN["estado"] == "corriendo" and 2 <= total < 100:
            tiempo += f", faltan ~{_fmt_t(activo * (100 - total) / total)}"
        cls, etq = {"corriendo": ("run", "Trabajando"), "pausa": ("pausa", "En pausa"),
                    "listo": ("ok", "Terminado"), "detenido": ("pausa", "Detenido"),
                    "error": ("err", "Con errores")}.get(RUN["estado"], ("run", ""))
        hecho_n = sum(1 for a in avs if a >= 99.9)
        msg = f'<div class="msg">{html.escape(RUN["msg"])}</div>' if RUN["msg"] else ""
        partes.append(
            f'<div class="resumen {cls}"><div class="fila"><span class="estado">{etq}</span>'
            f'<span class="total">{total:.0f}%</span>'
            f'<span class="cuenta">{hecho_n} de {len(vids)} listos</span>'
            f'<span class="tiempo">{tiempo}</span></div>'
            f'<div class="barra grande"><div class="relleno" style="width:{total:.1f}%"></div></div>{msg}</div>')

    # una fila por video o audio
    for v in vids:
        pct, actual = _avance_video(v, pasos)
        chips = []
        for p in (1, 2, 3):
            pp, _ = RUN["prog"].get((v["slug"], p), (None, ""))
            if pp is None:
                pp = 100.0 if v["hecho"][p] else 0.0
            if pp >= 99.9:
                c, txt = "ok", f"✓ {PASOS[p][1]}"
            elif pp > 0:
                c, txt = "run", f"{PASOS[p][1]} {pp:.0f}%"
            else:
                c, txt = "pend", PASOS[p][1]
            if p == 2 and v["auto"] and pp >= 99.9:
                txt += " (a revisar)"
            chips.append(f'<span class="paso {c}">{txt}</span>')
        trabajando = bool(actual and corriendo)
        listo = all(v["hecho"].values()) and not actual
        estado = "run" if trabajando and not RUN["pausado"] else ("pausa" if trabajando else ("ok" if listo else "pend"))
        det = ""
        if trabajando:
            det = f'<div class="detalle">{"En pausa: " if RUN["pausado"] else ""}{html.escape(actual[2] or "")}</div>'
        cancion = (f'<span class="musica" title="Partes cantadas que encontró">♪ {v["canciones"]}</span>'
                   if v.get("canciones") else "")
        partes.append(
            f'<div class="video {estado}"><div class="fila">'
            f'<span class="nombre" title="{html.escape(v["nombre"])}">{html.escape(v["nombre"])}</span>'
            f'<span class="tipo">{"audio" if v.get("audio") else "video"}</span>{cancion}'
            f'<span class="pct">{pct:.0f}%</span></div>'
            f'<div class="barra"><div class="relleno" style="width:{pct:.1f}%"></div></div>'
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


def traductores_instalados():
    env = APP_DIR / "env_traductor"
    return (env / "Scripts" / "python.exe").exists() or (env / "bin" / "python").exists()


def ejecutar(pasos, carpeta, idioma, voz, archivo_voz, contexto, forzar, origen=ORIGEN_AUTO, cantar=False,
             afinar=True, traductor=TRADUCTOR_BASE):
    carpeta = (carpeta or "").strip().strip('"')
    if not carpeta or not Path(carpeta).is_dir():
        raise gr.Error("Elegí primero la carpeta donde están los videos o audios.")
    if not any(p.suffix.lower() in EXT_MEDIOS for p in Path(carpeta).iterdir() if p.is_file()):
        raise gr.Error(f"En esa carpeta no hay videos ni audios ({EXTS_TXT}).")
    cod = COD_DE.get(idioma, "en")
    lista = list(pasos)
    if voz == VOZ_MUESTRA and 3 in lista and not archivo_voz and not muestra_lista(carpeta):
        raise gr.Error("Elegiste usar una muestra de voz propia: subí un audio o video con la voz (10-30 s).")
    clave_tr = TRADUCTORES_UI.get(traductor, "qwen3-4b")
    if 2 in lista and clave_tr != "qwen3-4b" and not traductores_instalados():
        raise gr.Error(f"Para usar {traductor.split(':')[0]}, en Pinokio tocá 'Instalar traductores' (una sola vez), "
                       "o elegí Qwen3-4B en Opciones avanzadas.")
    if not _ocupado.acquire(blocking=False):
        raise gr.Error("Ya hay un proceso corriendo. Esperá a que termine, o tocá Detener.")
    salida = deque(maxlen=500)
    try:
        guardar_ajustes(carpeta=carpeta, idioma=idioma, voz=voz, origen=origen, cantar=bool(cantar),
                        afinar=bool(afinar), traductor=traductor)
        proc = Path(carpeta) / "_proceso"
        proc.mkdir(parents=True, exist_ok=True)
        ajustes = {"carpeta_videos": carpeta, "idioma_codigo": cod,
                   "idioma_origen": ORIGENES_UI.get(origen, "auto"), "cantar": bool(cantar and motor_instalado()),
                   "canto_afinar": bool(afinar), "traductor": clave_tr}
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
                RUN["msg"] = f"Paso {paso} de 3, {PASOS[paso][1].lower()}: {PASOS[paso][2]}"
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


# ---------------------------------------------------------------- texto para tu voz
def leer_texto_archivo(archivo):
    """El texto de un .txt / .md / .srt / .docx cargado, para el cuadro de texto."""
    if not archivo:
        return gr.update()
    p = Path(archivo)
    try:
        if p.suffix.lower() == ".docx":
            import zipfile
            xml = zipfile.ZipFile(p).read("word/document.xml").decode("utf-8")
            pars = ["".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", par, re.S))
                    for par in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)]
            texto = "\n\n".join(html.unescape(x).strip() for x in pars if x.strip())
        else:
            crudo = p.read_bytes()
            texto = None
            for enc in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    texto = crudo.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if p.suffix.lower() == ".srt":  # solo lo que se dice, sin numeros ni tiempos
                texto = " ".join(l.strip() for l in texto.splitlines()
                                 if l.strip() and not l.strip().isdigit() and "-->" not in l)
    except Exception as e:
        raise gr.Error(f"No pude leer el archivo: {e}")
    return gr.update(value=(texto or "").strip())


def decir_texto(carpeta, voz, archivo_voz, texto, idioma_txt):
    """Texto -> mp3 con la voz clonada de la muestra (decir.py, en un proceso aparte)."""
    carpeta = (carpeta or "").strip().strip('"')
    texto = (texto or "").strip()
    if not texto:
        raise gr.Error("Escribí, pegá o cargá el texto que querés que diga tu voz.")
    if not carpeta or not Path(carpeta).is_dir():
        raise gr.Error("Elegí primero una carpeta: ahí se guarda tu muestra de voz y el audio que sale.")
    nueva = archivo_voz if (voz == VOZ_MUESTRA and archivo_voz) else None
    if not nueva and not muestra_lista(carpeta):
        raise gr.Error("Falta tu muestra de voz: en el punto 3 (Voz) elegí 'Una muestra de voz propia' y subí "
                       "10-30 s de vos hablando.")
    if not _ocupado.acquire(blocking=False):
        raise gr.Error("Ya hay un proceso corriendo. Esperá a que termine, o tocá Detener.")
    salida = deque(maxlen=500)
    try:
        guardar_ajustes(carpeta=carpeta, texto_idioma=idioma_txt)
        proc = Path(carpeta) / "_proceso"
        tdir = proc / "textos"
        tdir.mkdir(parents=True, exist_ok=True)
        tx = tdir / "texto.txt"
        tx.write_text(texto + "\n", encoding="utf-8")
        cod = IDIOMAS_TEXTO.get(idioma_txt, "auto")
        aj_path = tdir / "ajustes_texto.json"
        aj_path.write_text(json.dumps({"carpeta_videos": carpeta, "idioma_codigo": cod if cod in IDIOMAS else "en",
                                       "voz_muestra": str(proc / "voz_muestra"), "texto_archivo": str(tx),
                                       "texto_idioma": cod}, ensure_ascii=False, indent=2), encoding="utf-8")
        RUN["detener"] = False
        for k in (("voz", "voz"), ("texto", "decir")):
            RUN["prog"].pop(k, None)
        pasos = [("voz_muestra.py", [f"--archivo={nueva}"], "Preparando tu muestra de voz", ("voz", "voz"))] \
            if nueva else []
        pasos.append(("decir.py", [], "Diciendo el texto con tu voz", ("texto", "decir")))
        for script, extra, msg, clave in pasos:
            if os.environ.get("DOBLAJE_SIMULAR"):
                extra = extra + ["--simular"]  # solo para pruebas
            salida.append("")
            salida.append(f"=== {msg} ===")
            hilo = _correr(script, [f"--config={aj_path}", *extra], salida)
            while hilo.is_alive() or _proc.poll() is None:
                pct, det = RUN["prog"].get(clave, (None, ""))
                yield (f"{msg}…" + (f" {det} ({pct:.0f}%)" if pct is not None else "")), _registro(salida), \
                    gr.update()
                time.sleep(1.0)
            hilo.join(timeout=2)
            if RUN["detener"]:
                yield "Detenido.", _registro(salida), gr.update()
                return
            if _proc.returncode != 0:
                yield ("Terminó con errores: el detalle está en 'Detalle técnico' y en _proceso/log.txt."), \
                    _registro(salida), gr.update()
                return
        info = json.loads((tdir / "ultimo.json").read_text(encoding="utf-8"))
        final = Path(info["archivo"])
        copia = Path(tempfile.mkdtemp(prefix="texto_")) / final.name  # gradio solo sirve archivos de su carpeta
        shutil.copyfile(final, copia)
        flojas = info.get("flojas") or []
        nota = (f" Ojo: {len(flojas)} frase(s) no salieron del todo claras: escuchalas; si alguna está mal, cambiale "
                "algo y generá de nuevo (solo se rehacen las que cambian).") if flojas else ""
        yield (f"Listo: **{final.name}**, en salida/textos.{nota}", _registro(salida),
               gr.update(value=str(copia), visible=True))
    finally:
        _ocupado.release()


# ---------------------------------------------------------------- estilo
# Cabina de doblaje: fondo de fieltro acustico, verde pino para lo que esta listo o se puede tocar, y ambar
# de vumetro solo para lo que esta trabajando. Letra Lexend (hecha para leer con menos esfuerzo), local.
COLORES = {
    #             claro      oscuro
    "fondo": ("#E8EEEA", "#131C1A"),
    "consola": ("#FAFCFA", "#1B2725"),
    "campo": ("#FFFFFF", "#15201E"),
    "tinta": ("#1C2B28", "#E2EBE7"),
    "suave": ("#5B6D68", "#93A6A0"),
    "linea": ("#D3DDD8", "#2A3936"),
    "pista": ("#DCE4E0", "#263431"),
    "onda": ("#AEBDB7", "#3C4F4A"),
    "pino": ("#1E6B58", "#4BB497"),
    "pino_osc": ("#17594A", "#3E9C82"),
    "pino_claro": ("#D9EAE3", "#1F3D35"),
    "sobre_pino": ("#FFFFFF", "#0F1E1A"),
    "ambar": ("#E39B2D", "#F2B54A"),
    "ambar_claro": ("#F8E6C6", "#3D3220"),
    "ambar_texto": ("#8A5A10", "#F2B54A"),
    "rojo": ("#C8463D", "#EF6A5F"),
    "rojo_claro": ("#F8E1DE", "#3A2321"),
}


def _vars(i):
    return "".join(f"--c-{k.replace('_', '-')}: {v[i]}; " for k, v in COLORES.items())


def _css_fuente():
    try:
        b64 = base64.b64encode((APP_DIR / "fuentes" / "lexend.woff2").read_bytes()).decode()
    except OSError:
        return ""
    return ("@font-face {font-family: 'Lexend'; font-style: normal; font-weight: 100 900; font-display: swap; "
            f"src: url(data:font/woff2;base64,{b64}) format('woff2');}}\n")


def _onda():
    """La firma de la cabecera: la misma frase dicha dos veces, con el mismo ritmo y otra voz."""
    import math
    import random
    rnd = random.Random(11)
    n, ancho, paso, alto = 52, 7, 11, 92
    altos = []
    for i in range(n // 2):
        env = 0.28 + 0.72 * abs(math.sin(i * 0.62 + 0.4)) * (0.55 + 0.45 * math.sin(i * 0.21 + 1.2))
        altos.append(max(6.0, alto * min(1.0, env * (0.8 + 0.4 * rnd.random()))))
    orig, dob = [], []
    for i in range(n):
        j = i % (n // 2)
        h = altos[j] if i < n // 2 else max(6.0, min(alto, altos[j] * (0.86 + 0.28 * rnd.random())))
        r = f'<rect x="{i * paso}" y="{(alto - h) / 2 + 8:.1f}" width="{ancho}" height="{h:.1f}" rx="3.5"'
        (orig if i < n // 2 else dob).append(r + ("/>" if i < n // 2 else f' style="--i:{j}"/>'))
    x = (n // 2) * paso - (paso - ancho) / 2
    w = n * paso - (paso - ancho)
    return (f'<svg class="onda" viewBox="-2 0 {w + 4} {alto + 16}" role="img" '
            f'aria-label="La misma frase con el mismo ritmo, en otra voz">'
            f'<g class="orig">{"".join(orig)}</g><g class="doblada">{"".join(dob)}</g>'
            f'<line class="cabezal" x1="{x}" y1="2" x2="{x}" y2="{alto + 14}"/></svg>')


CABECERA = (
    '<header id="cabecera"><div class="cab-texto"><h1>Doblaje con voz clonada</h1>'
    '<p>Traducí y doblá videos o audios al inglés, alemán, ruso o castellano con la voz de quien habla, '
    'o hacé que tu voz diga cualquier texto. Todo corre en tu compu, sin cuentas ni API key.</p></div>'
    + _onda() + '</header>')

CSS = _css_fuente() + ":root {" + _vars(0) + "}\n.dark {" + _vars(1) + "}\n" + """
.gradio-container {max-width: 1240px !important; margin: auto !important;}
footer {display: none !important;}

/* cabecera */
#cabecera {display: flex; align-items: flex-end; justify-content: space-between; gap: 36px; flex-wrap: wrap;
           padding: 30px 6px 20px;}
#cabecera h1 {margin: 0; font-size: 40px; line-height: 1.06; font-weight: 640; letter-spacing: -0.025em;
              color: var(--c-tinta);}
#cabecera p {margin: 12px 0 0; max-width: 58ch; font-size: 16px; line-height: 1.6; color: var(--c-suave);}
#cabecera .onda {width: min(400px, 100%); height: auto; flex-shrink: 0; margin-bottom: 4px;}
.onda .orig rect {fill: var(--c-onda);}
.onda .doblada rect {fill: var(--c-pino); transform-box: fill-box; transform-origin: center;
                     animation: crecer .75s cubic-bezier(.2, .8, .2, 1) both;
                     animation-delay: calc(var(--i) * 28ms + 200ms);}
.onda .cabezal {stroke: var(--c-ambar); stroke-width: 3; stroke-linecap: round;}
@keyframes crecer {from {transform: scaleY(.05);} to {transform: scaleY(1);}}

/* tarjetas y secciones */
.tarjeta {background: var(--c-consola) !important; border: 1px solid var(--c-linea) !important;
          border-radius: 18px !important; padding: 22px 22px 20px !important; box-shadow: none !important;}
.seccion {display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 15.5px;
          color: var(--c-tinta); margin: 8px 0 -2px;}
.seccion .num {display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 50%;
               background: var(--c-pino); color: var(--c-sobre-pino); font-size: 13px; font-weight: 600;}
.nota, .nota p {font-size: 14.5px !important; line-height: 1.55; color: var(--c-suave) !important;}
#btn-todo {font-size: 18px !important; padding: 15px !important; border-radius: 14px !important;
           margin-top: 6px;}
#pasos-sueltos {flex-wrap: nowrap !important; gap: 8px !important;}
#pasos-sueltos button {color: var(--c-suave) !important; background: transparent !important;}
#btn-ref {max-width: 44px !important; min-width: 44px !important; font-size: 18px !important;}

/* progreso */
#panel {max-height: 660px; overflow-y: auto; padding-right: 4px;}
.vacio {padding: 56px 24px; text-align: center; color: var(--c-suave); font-size: 15px; line-height: 1.6;}
.fila {display: flex; align-items: center; gap: 12px;}
.resumen {padding: 16px 18px 14px; border-radius: 14px; margin-bottom: 14px; background: var(--c-fondo);}
.resumen .estado {font-size: 12.5px; font-weight: 600; padding: 3px 11px; border-radius: 999px;
                  background: var(--c-pino-claro); color: var(--c-pino);}
.resumen.run .estado, .resumen.pausa .estado {background: var(--c-ambar-claro); color: var(--c-ambar-texto);}
.resumen.err .estado {background: var(--c-rojo-claro); color: var(--c-rojo);}
.resumen .total {font-size: 26px; font-weight: 640; letter-spacing: -0.015em; font-variant-numeric: tabular-nums;}
.resumen .cuenta {font-size: 14px; color: var(--c-suave);}
.resumen .tiempo {margin-left: auto; font-size: 13.5px; color: var(--c-suave); white-space: nowrap;
                  font-variant-numeric: tabular-nums;}
.resumen .msg {margin-top: 10px; font-size: 14px; line-height: 1.5; color: var(--c-suave);}
.barra {height: 8px; border-radius: 999px; background: var(--c-pista); overflow: hidden; margin-top: 10px;}
.barra.grande {height: 12px;}
.relleno {height: 100%; border-radius: 999px; background: var(--c-pino); transition: width .8s ease;}
.run .relleno {background-color: var(--c-ambar);
               background-image: repeating-linear-gradient(135deg, rgba(255,255,255,.28) 0 7px, transparent 7px 14px);
               background-size: 19.8px 19.8px; animation: avanzar 1.1s linear infinite;}
.pausa .relleno {background: var(--c-ambar);}
.err .relleno {background: var(--c-rojo);}
@keyframes avanzar {from {background-position: 0 0;} to {background-position: 19.8px 0;}}
.video {padding: 12px 14px 12px 15px; margin-bottom: 8px; border-radius: 12px; background: var(--c-consola);
        border: 1px solid var(--c-linea); border-left: 4px solid var(--c-linea);}
.video.ok {border-left-color: var(--c-pino);}
.video.run, .video.pausa {border-left-color: var(--c-ambar);}
.video .nombre {flex: 1; min-width: 0; font-weight: 500; font-size: 14.5px; color: var(--c-tinta);
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.video .tipo {font-size: 12.5px; color: var(--c-suave);}
.musica {font-size: 12.5px; font-weight: 600; color: var(--c-pino); background: var(--c-pino-claro);
         border-radius: 999px; padding: 1px 9px;}
.video .pct {min-width: 3em; text-align: right; font-weight: 600; font-size: 14px; color: var(--c-suave);
             font-variant-numeric: tabular-nums;}
.video.ok .pct {color: var(--c-pino);}
.video.run .pct, .video.pausa .pct {color: var(--c-ambar-texto);}
.pasos {display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px;}
.paso {font-size: 12.5px; padding: 2px 10px; border-radius: 999px; background: var(--c-fondo); color: var(--c-suave);}
.paso.ok {background: var(--c-pino-claro); color: var(--c-pino);}
.paso.run {background: var(--c-ambar-claro); color: var(--c-ambar-texto); font-weight: 600;}
.detalle {margin-top: 8px; font-size: 13.5px; color: var(--c-suave);}

.subir {border: 1.5px dashed var(--c-linea) !important; border-radius: 12px !important;
        background: var(--c-fondo) !important; margin-top: 4px;}
.subir:hover {border-color: var(--c-pino) !important;}
.subir .or {opacity: .7;}

/* texto para tu voz */
#tarjeta-texto {margin-top: 16px;}
#texto-voz textarea {font-size: 16px !important; line-height: 1.65 !important;}
#estado-texto p {font-size: 15px; margin: 4px 0;}

@media (prefers-reduced-motion: reduce) {
  .onda .doblada rect, .run .relleno {animation: none !important;}
  .relleno {transition: none;}
}
@media (max-width: 720px) {
  #cabecera h1 {font-size: 31px;}
  #cabecera {padding-top: 18px;}
  .resumen .tiempo {display: none;}
}
"""


def crear_tema():
    c = {k: v[0] for k, v in COLORES.items()}
    o = {k: v[1] for k, v in COLORES.items()}
    tema = gr.themes.Base(font=["Lexend", "Segoe UI", "system-ui", "sans-serif"],
                          font_mono=["Cascadia Mono", "Consolas", "ui-monospace", "monospace"],
                          radius_size=gr.themes.sizes.radius_md, text_size=gr.themes.sizes.text_md)
    pares = {
        "body_background_fill": "fondo", "body_text_color": "tinta", "body_text_color_subdued": "suave",
        "background_fill_primary": "consola", "background_fill_secondary": "fondo",
        "border_color_primary": "linea", "border_color_accent": "pino", "border_color_accent_subdued": "pino_claro",
        "color_accent": "pino", "color_accent_soft": "pino_claro",
        "link_text_color": "pino", "link_text_color_hover": "pino_osc", "link_text_color_visited": "pino",
        "block_background_fill": "consola", "block_border_color": "linea", "block_title_text_color": "tinta",
        "block_label_text_color": "suave", "block_info_text_color": "suave",
        "input_background_fill": "campo", "input_background_fill_focus": "campo", "input_border_color": "linea",
        "input_border_color_focus": "pino", "input_border_color_hover": "suave", "input_placeholder_color": "suave",
        "checkbox_label_background_fill": "campo", "checkbox_label_background_fill_hover": "pino_claro",
        "checkbox_label_background_fill_selected": "pino", "checkbox_label_border_color": "linea",
        "checkbox_label_border_color_hover": "pino", "checkbox_label_border_color_selected": "pino",
        "checkbox_label_text_color": "tinta", "checkbox_label_text_color_selected": "sobre_pino",
        "checkbox_background_color": "campo", "checkbox_background_color_selected": "pino",
        "checkbox_border_color": "suave", "checkbox_border_color_selected": "pino",
        "checkbox_border_color_focus": "pino",
        "button_primary_background_fill": "pino", "button_primary_background_fill_hover": "pino_osc",
        "button_primary_border_color": "pino", "button_primary_border_color_hover": "pino_osc",
        "button_primary_text_color": "sobre_pino", "button_primary_text_color_hover": "sobre_pino",
        "button_secondary_background_fill": "consola", "button_secondary_background_fill_hover": "pino_claro",
        "button_secondary_border_color": "linea", "button_secondary_border_color_hover": "pino",
        "button_secondary_text_color": "tinta", "button_secondary_text_color_hover": "tinta",
        "button_cancel_background_fill": "consola", "button_cancel_background_fill_hover": "rojo_claro",
        "button_cancel_border_color": "rojo", "button_cancel_border_color_hover": "rojo",
        "button_cancel_text_color": "rojo", "button_cancel_text_color_hover": "rojo",
        "loader_color": "ambar", "slider_color": "pino", "accordion_text_color": "tinta",
        "panel_background_fill": "consola", "panel_border_color": "linea",
    }
    import inspect
    validos = set(inspect.signature(gr.themes.Base.set).parameters)  # cambian un poco entre versiones de gradio
    ajustes = {}
    for var, color in pares.items():
        ajustes[var] = c[color]
        ajustes[var + "_dark"] = o[color]
    ajustes.update(
        body_text_size="15px", block_border_width="0px", block_shadow="none", block_padding="6px 0",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_border_width="0px", block_label_text_size="14px", block_label_text_weight="500",
        block_title_text_size="14.5px", block_title_text_weight="500", block_info_text_size="13px",
        input_radius="10px", input_border_width="1px", input_shadow="none", input_shadow_focus="0 0 0 3px rgba(30,107,88,.16)",
        input_shadow_focus_dark="0 0 0 3px rgba(75,180,151,.22)", input_text_size="15px",
        checkbox_label_shadow="none", checkbox_label_shadow_hover="none", checkbox_label_padding="8px 14px",
        checkbox_label_text_size="15px", checkbox_label_border_width="1px", checkbox_shadow="none",
        button_border_width="1px", button_transform_hover="none", button_transform_active="scale(.98)",
        button_large_radius="12px", button_medium_radius="10px", button_small_radius="8px",
        button_large_text_size="17px", button_large_text_weight="600", button_medium_text_weight="500",
        button_small_text_weight="500", button_primary_shadow="none", button_primary_shadow_hover="none",
        button_secondary_shadow="none", button_secondary_shadow_hover="none", button_cancel_shadow="none",
        button_cancel_shadow_hover="none", layout_gap="14px", panel_border_width="0px",
        shadow_drop="none", shadow_drop_lg="none", shadow_spread="0px",
        container_radius="14px", embed_radius="14px",
    )
    return tema.set(**{k: v for k, v in ajustes.items() if k in validos})


# ---------------------------------------------------------------- interfaz
def _titulo(texto, num=None):
    marca = f'<span class="num">{num}</span>' if num else ""
    return gr.HTML(f'<div class="seccion">{marca}{texto}</div>')


def armar():
    aj = cargar_ajustes()
    with gr.Blocks(title="Doblaje con voz clonada") as demo:
        gr.HTML(CABECERA)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="tarjeta"):
                _titulo("Carpeta con los videos o audios", 1)
                with gr.Row():
                    carpeta = gr.Textbox(value=aj["carpeta"], show_label=False, scale=5,
                                         placeholder="C:\\Users\\...\\Desktop\\mis videos")
                    b_elegir = gr.Button("Elegir…", scale=1, min_width=100)
                ultima = gr.State(aj["carpeta"])
                _titulo("Idioma del doblaje", 2)
                idioma = gr.Radio(OPC_IDIOMA, value=aj["idioma"], show_label=False)
                _titulo("Voz", 3)
                voz = gr.Radio([VOZ_VIDEO, VOZ_MUESTRA], value=aj["voz"], show_label=False)
                archivo_voz = gr.File(label="Muestra de voz",
                                      file_types=["audio", "video"], type="filepath", height=150,
                                      visible=aj["voz"] == VOZ_MUESTRA, elem_classes="subir")
                info_voz = gr.Markdown(visible=aj["voz"] == VOZ_MUESTRA, elem_classes="nota")
                hay_canto = motor_instalado()
                cantar = gr.Checkbox(
                    label="Cantar las canciones: detecta las partes cantadas y las canta en el idioma del doblaje, "
                          "con la misma melodía",
                    value=bool(aj.get("cantar", True)) and hay_canto, interactive=hay_canto,
                    info=None if hay_canto else "Para usarlo, en Pinokio tocá 'Instalar canto' (una sola vez).")
                with gr.Accordion("Opciones avanzadas", open=False):
                    origen = gr.Dropdown(list(ORIGENES_UI), value=aj["origen"],
                                         label="Idioma en que hablan los videos o audios originales")
                    contexto = gr.Textbox(label="De qué trata (ayuda a traducir mejor: quién habla y de qué)",
                                          value=aj.get("contexto", ""), lines=2,
                                          placeholder="Ej.: habla un psicólogo, en primera persona, sobre arte y "
                                                      "salud mental. Se guarda para esta carpeta.")
                    afinar = gr.Checkbox(label="Afinar el canto con autotune (solo si falta el conversor de voz "
                                               "cantada)", value=bool(aj.get("afinar", True)), interactive=hay_canto)
                    traductor = gr.Dropdown(list(TRADUCTORES_UI), value=aj["traductor"], label="Traductor",
                                            info="Los dos nuevos se instalan una vez desde Pinokio ('Instalar "
                                                 "traductores'). Al cambiar de traductor se rehacen las traducciones "
                                                 "automáticas; las revisadas no se tocan.")
                    forzar = gr.Checkbox(label="Rehacer las traducciones automáticas (las revisadas no se tocan)",
                                         value=False)
                b_todo = gr.Button("▶  Hacer todo", variant="primary", elem_id="btn-todo")
                with gr.Row(elem_id="pasos-sueltos"):
                    b1 = gr.Button("Solo preparar", size="sm", min_width=60)
                    b2 = gr.Button("Solo traducir", size="sm", min_width=60)
                    b3 = gr.Button("Solo doblar", size="sm", min_width=60)
                with gr.Row():
                    b_pausa = gr.Button("⏸ Pausar", variant="secondary")
                    b_stop = gr.Button("⏹ Detener", variant="stop", elem_id="btn-stop")
                b_salida = gr.Button("Abrir la carpeta de resultados", variant="secondary")
            with gr.Column(scale=7, elem_classes="tarjeta"):
                with gr.Row(equal_height=True):
                    _titulo("Progreso")
                    b_ref = gr.Button("↻", size="sm", min_width=44, scale=0, elem_id="btn-ref")
                tablero = gr.HTML(panel(aj["carpeta"], aj["idioma"]), elem_id="panel")
                with gr.Accordion("Detalle técnico", open=False):
                    registro = gr.Textbox(show_label=False, lines=16, max_lines=16, autoscroll=True,
                                          interactive=False)
        with gr.Column(elem_classes="tarjeta", elem_id="tarjeta-texto"):
            _titulo("Texto para tu voz")
            gr.Markdown("Escribí, pegá o cargá un texto y lo dice tu voz clonada (la muestra de voz de la carpeta "
                        "elegida, en el punto 3). Sale un mp3 en la carpeta salida/textos.", elem_classes="nota")
            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    texto = gr.Textbox(show_label=False, lines=9, max_lines=30, elem_id="texto-voz",
                                       placeholder="Escribí acá lo que querés que diga tu voz. Dejá una línea en "
                                                   "blanco entre párrafos para que haga una pausa más larga.")
                with gr.Column(scale=3, min_width=250):
                    archivo_txt = gr.File(label="O cargá un archivo (.txt, .docx, .md, .srt)",
                                          file_types=[".txt", ".docx", ".md", ".srt"], type="filepath", height=150,
                                          elem_classes="subir")
                    idioma_txt = gr.Dropdown(list(IDIOMAS_TEXTO), value=aj["texto_idioma"],
                                             label="Idioma en que está escrito")
                    b_decir = gr.Button("Generar audio con mi voz", variant="primary")
                    b_decir_stop = gr.Button("⏹ Detener", variant="stop", size="sm")
            estado_txt = gr.Markdown("", elem_id="estado-texto")
            audio_txt = gr.Audio(label="Resultado", type="filepath", interactive=False, visible=False)

        def ver_voz(v, c):
            m = muestra_lista(c)
            txt = ("10 a 30 s de una sola persona hablando, sin música. "
                   + (f"Ya hay una preparada en esta carpeta: **{m}** (subí otra para cambiarla)."
                      if m else "Todavía no hay ninguna en esta carpeta."))
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
        demo.load(ver_voz, [voz, carpeta], [archivo_voz, info_voz])

        def hacer(pasos):
            def correr_pasos(carpeta_, idioma_, voz_, archivo_, contexto_, forzar_, origen_, cantar_, afinar_,
                             traductor_):
                yield from ejecutar(pasos, carpeta_, idioma_, voz_, archivo_, contexto_, forzar_, origen_, cantar_,
                                    afinar_, traductor_)
            return correr_pasos

        entradas = [carpeta, idioma, voz, archivo_voz, contexto, forzar, origen, cantar, afinar, traductor]
        for boton, pasos in ((b_todo, (1, 2, 3)), (b1, (1,)), (b2, (2,)), (b3, (3,))):
            boton.click(hacer(pasos), entradas, [tablero, registro, b_pausa])
        b_pausa.click(pausar_continuar, None, b_pausa, concurrency_limit=None, queue=False)
        b_stop.click(detener, None, b_pausa, concurrency_limit=None, queue=False)
        b_salida.click(abrir_salida, carpeta, None, queue=False)
        archivo_txt.upload(leer_texto_archivo, archivo_txt, texto)
        b_decir.click(decir_texto, [carpeta, voz, archivo_voz, texto, idioma_txt], [estado_txt, registro, audio_txt])
        b_decir_stop.click(detener, None, b_pausa, concurrency_limit=None, queue=False)
    return demo


if __name__ == "__main__":
    armar().queue(default_concurrency_limit=2).launch(server_name="127.0.0.1", inbrowser=False, theme=crear_tema(),
                                                      css=CSS)
