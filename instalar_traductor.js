module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    // 1) entorno propio para los traductores nuevos (transformers 5; no se mezcla con el del doblaje)
    {
      method: "shell.run",
      params: {
        venv: "env_traductor",
        path: "app",
        message: [
          "uv pip install -r traductor/requirements_traductor.txt"
        ]
      }
    },
    // 2) torch con CUDA al final, como en install.js
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env_traductor",
          path: "app"
        }
      }
    },
    // 3) prueba rapida: transformers, CUDA y bitsandbytes
    {
      method: "shell.run",
      params: {
        venv: "env_traductor",
        path: "app",
        env: { PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
        message: [
          "python traductor/servidor.py --probar"
        ]
      }
    },
    {
      method: "input",
      params: {
        title: "Traductores instalados",
        description: "Listo. Volve a Iniciar la app y elegi el traductor en Opciones avanzadas. La primera vez que uses cada uno se baja su modelo (Hy-MT2-7B unos 16 GB, Qwen3.5-9B unos 19 GB)."
      }
    }
  ]
}
