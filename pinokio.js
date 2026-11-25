const fs = require("fs")
const path = require("path")

// instalado = el entorno tiene qwen_tts (no alcanza con que exista la carpeta env)
const instalado = () => {
  const venv = path.join(__dirname, "app", "env")
  const cands = [path.join(venv, "Lib", "site-packages", "qwen_tts")]
  try {
    for (const d of fs.readdirSync(path.join(venv, "lib"))) {
      cands.push(path.join(venv, "lib", d, "site-packages", "qwen_tts"))
    }
  } catch (e) {}
  return cands.some((p) => fs.existsSync(p))
}

module.exports = {
  version: "3.7",
  title: "Doblaje Gregorio",
  description: "Traduce y dobla videos o audios (ingles, aleman, ruso o espanol) con voz clonada y subtitulos, todo local: Demucs + Whisper + Qwen",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const r = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      reset: info.running("reset.js"),
    }
    if (r.install) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Instalando...", href: "install.js" }]
    }
    if (r.reset) {
      return [{ default: true, icon: "fa-solid fa-terminal", text: "Borrando entorno...", href: "reset.js" }]
    }
    if (!instalado()) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Instalar", href: "install.js" }]
    }
    if (r.start) {
      const local = info.local("start.js")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "<div><strong>Abrir interfaz</strong><div>Carpeta, idioma, voz y Hacer todo</div></div>",
          href: local.url,
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        }]
      }
      return [{ default: true, icon: "fa-solid fa-terminal", text: "Iniciando...", href: "start.js" }]
    }
    return [{
      default: true,
      icon: "fa-solid fa-power-off",
      text: "<div><strong>Iniciar</strong><div>Abre la interfaz del doblaje</div></div>",
      href: "start.js",
    }, {
      icon: "fa-solid fa-plug",
      text: "Reinstalar",
      href: "install.js",
    }, {
      icon: "fa-regular fa-circle-xmark",
      text: "Borrar entorno",
      href: "reset.js",
      confirm: "¿Borrar el entorno de Python? (no toca tus videos)"
    }]
  }
}
