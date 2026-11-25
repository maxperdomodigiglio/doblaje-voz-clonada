module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "input",
      params: {
        title: "Instalacion completa",
        description: "Toca 'Iniciar' y despues 'Abrir interfaz'. La primera vez se descargan los modelos (Whisper, Demucs, Qwen3-TTS y el traductor local, unos 16 GB en total)."
      }
    }
  ]
}
