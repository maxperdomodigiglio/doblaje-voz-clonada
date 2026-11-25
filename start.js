module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: { PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1" },
        message: [
          "uv pip install num2words pyworld",
          "python ui.py"
        ],
        on: [{
          event: "/http:\\/\\/[0-9.:]+/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
