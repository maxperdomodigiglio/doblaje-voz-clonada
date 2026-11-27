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

// canto instalado = entorno env_canto + codigo de YingMusic-Singer-Plus
const cantoInstalado = () => {
  const app = path.join(__dirname, "app")
  const py = [path.join(app, "env_canto", "Scripts", "python.exe"), path.join(app, "env_canto", "bin", "python")]
  return py.some((p) => fs.existsSync(p)) &&
    fs.existsSync(path.join(app, "canto", "YingMusic-Singer-Plus", "src", "YingMusicSinger"))
}

// conversor de voz cantada (YingMusic-SVC): viene con el canto desde la v15
const conversorInstalado = () =>
  fs.existsSync(path.join(__dirname, "app", "canto", "YingMusic-SVC", "modules", "rmvpe.py"))

const itemCanto = () => cantoInstalado() ? (conversorInstalado() ? {
  icon: "fa-solid fa-music",
  text: "<div><strong>Reinstalar canto</strong><div>YingMusic-Singer + conversor de voz (canciones)</div></div>",
  href: "instalar_canto.js",
} : {
  icon: "fa-solid fa-music",
  text: "<div><strong>Actualizar canto</strong><div>Nuevo: la voz cae en las notas de la cancion (~3 GB)</div></div>",
  href: "instalar_canto.js",
}) : {
  icon: "fa-solid fa-music",
  text: "<div><strong>Instalar canto</strong><div>Para cantar las canciones traducidas (~8 GB)</div></div>",
  href: "instalar_canto.js",
}

// traductores nuevos (Qwen3.5-9B y Hy-MT2-7B) = entorno env_traductor
const traductoresInstalados = () => {
  const app = path.join(__dirname, "app")
  return [path.join(app, "env_traductor", "Scripts", "python.exe"), path.join(app, "env_traductor", "bin", "python")]
    .some((p) => fs.existsSync(p))
}

const itemTraductor = () => traductoresInstalados() ? {
  icon: "fa-solid fa-language",
  text: "<div><strong>Reinstalar traductores</strong><div>Qwen3.5-9B y Hy-MT2-7B</div></div>",
  href: "instalar_traductor.js",
} : {
  icon: "fa-solid fa-language",
  text: "<div><strong>Instalar traductores</strong><div>Qwen3.5-9B y Hy-MT2-7B, para traducir mejor (~3 GB + modelos)</div></div>",
  href: "instalar_traductor.js",
}

module.exports = {
  version: "3.7",
  title: "Doblaje Gregorio",
  description: "Traduce y dobla videos o audios (ingles, aleman, ruso o espanol) con voz clonada y subtitulos, todo local: Demucs + Whisper + Qwen",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const r = {
      install: info.running("install.js"),
      canto: info.running("instalar_canto.js"),
      traductor: info.running("instalar_traductor.js"),
      start: info.running("start.js"),
      reset: info.running("reset.js"),
    }
    if (r.install) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Instalando...", href: "install.js" }]
    }
    if (r.canto) {
      return [{ default: true, icon: "fa-solid fa-music", text: "Instalando el canto...", href: "instalar_canto.js" }]
    }
    if (r.traductor) {
      return [{ default: true, icon: "fa-solid fa-language", text: "Instalando los traductores...", href: "instalar_traductor.js" }]
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
        }, itemCanto(), itemTraductor()]
      }
      return [{ default: true, icon: "fa-solid fa-terminal", text: "Iniciando...", href: "start.js" }]
    }
    return [{
      default: true,
      icon: "fa-solid fa-power-off",
      text: "<div><strong>Iniciar</strong><div>Abre la interfaz del doblaje</div></div>",
      href: "start.js",
    }, itemCanto(), itemTraductor(), {
      icon: "fa-solid fa-plug",
      text: "Reinstalar",
      href: "install.js",
    }, {
      icon: "fa-regular fa-circle-xmark",
      text: "Borrar entorno",
      href: "reset.js",
      confirm: "¿Borrar los entornos de Python (doblaje, canto y traductores)? (no toca tus videos)"
    }]
  }
}
