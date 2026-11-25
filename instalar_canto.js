module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    // 1) codigo de YingMusic-Singer-Plus (version fija)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: { PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1" },
        message: [
          "python canto/instalar.py"
        ]
      }
    },
    // 2) entorno propio para el canto (no se mezcla con el del doblaje); torch con CUDA al final, como en install.js
    {
      method: "shell.run",
      params: {
        venv: "env_canto",
        path: "app",
        message: [
          "uv pip install -r canto/requirements_canto.txt"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env_canto",
          path: "app"
        }
      }
    },
    // 3) prueba rapida: que espeak y el tokenizador anden (muestra fonemas en cada idioma)
    {
      method: "shell.run",
      params: {
        venv: "env_canto",
        path: "app",
        env: { PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
        message: [
          "python canto/canto_motor.py --fonemas canto/YingMusic-Singer-Plus"
        ]
      }
    },
    {
      method: "input",
      params: {
        title: "Canto instalado",
        description: "Listo. Volve a Iniciar la app y deja tildado 'Cantar las canciones'. La primera vez que cante baja el modelo de canto (~3 GB)."
      }
    }
  ]
}
