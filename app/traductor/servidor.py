"""Traductores nuevos (Qwen3.5-9B y Hy-MT2-7B), en su propio entorno (env_traductor, transformers 5).

Lo lanza paso2_traducir.py y le habla por stdin/stdout, una linea JSON por pedido:

    {"op": "chat", "modelo": "tencent/Hy-MT2-7B", "mensajes": [{"role": "user", "content": "..."}],
     "max_new_tokens": 300, "temperatura": 0.0, "semilla": 123}
    -> {"ok": true, "texto": "..."}
    {"op": "fin"}

Los modelos se cargan la primera vez que se piden y quedan en memoria (en 4 bits si la placa tiene menos de
22 GB). Todo lo que no es respuesta (avisos, descargas) va a stderr, que se ve en el registro de la app.
"""
import json
import os
import sys
import time
from pathlib import Path

PROTO = sys.stdout
sys.stdout = sys.stderr  # nada de las librerias se mezcla con las respuestas
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

AQUI = Path(__file__).resolve().parent
CACHE = str(AQUI.parent / "modelos" / "hf")
RUTAS = json.loads(os.environ.get("TRADUCTOR_RUTAS", "{}"))  # solo para pruebas: nombre -> carpeta local

# como se usa cada modelo (lo que recomienda cada tarjeta de Hugging Face)
PERFILES = {
    "tencent/Hy-MT2-7B": {"sistema": False, "muestrear": True,
                          "muestreo": dict(temperature=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05)},
    "Qwen/Qwen3.5-9B": {"sistema": True, "sin_pensar": True,
                        "muestreo": dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)},
    "Qwen/Qwen3-4B-Instruct-2507": {"sistema": True, "muestreo": dict(temperature=0.4, top_p=0.9,
                                                                      repetition_penalty=1.05)},
}


def log(msg):
    print(f"[traductor] {msg}", file=sys.stderr, flush=True)


class Modelos:
    def __init__(self):
        import torch
        self.torch = torch
        self.cargados = {}
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if self.dev == "cuda" else 0
        log(f"placa: {torch.cuda.get_device_name(0) if self.dev == 'cuda' else 'sin GPU'}"
            + (f", {self.vram:.0f} GB" if self.vram else ""))

    def cargar(self, nombre):
        if nombre in self.cargados:
            return self.cargados[nombre]
        torch = self.torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        completo = self.vram >= 22 or os.environ.get("TRADUCTOR_COMPLETO")
        modo = "sin GPU, lento" if self.dev != "cuda" else ("completo" if completo else "en 4 bits")
        log(f"cargando {nombre} ({modo}; la primera vez se descarga, entre 8 y 19 GB)...")
        kw = dict(cache_dir=CACHE)
        ruta = RUTAS.get(nombre, nombre)
        try:
            tok = AutoTokenizer.from_pretrained(ruta, **kw)
        except Exception as e:  # algunos tokenizadores vienen con codigo propio en el repo
            log(f"tokenizador: {type(e).__name__}: {e}; pruebo con el codigo del repo")
            tok = AutoTokenizer.from_pretrained(ruta, trust_remote_code=True, **kw)
        if self.dev != "cuda":
            mkw = dict(dtype=torch.float32)
        elif completo:
            mkw = dict(dtype=torch.bfloat16, device_map="cuda")
        else:
            from transformers import BitsAndBytesConfig
            mkw = dict(device_map="cuda", dtype=torch.bfloat16,
                       quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                              bnb_4bit_use_double_quant=True,
                                                              bnb_4bit_compute_dtype=torch.bfloat16))
        modelo = AutoModelForCausalLM.from_pretrained(ruta, **kw, **mkw)
        modelo.eval()
        uso = torch.cuda.memory_allocated() / 2**30 if self.dev == "cuda" else 0
        log(f"{nombre} listo en {time.time() - t0:.0f} s" + (f" ({uso:.1f} GB en la placa)" if uso else ""))
        self.cargados[nombre] = (tok, modelo)
        return tok, modelo

    def chat(self, nombre, mensajes, max_new_tokens=400, temperatura=0.0, semilla=None):
        torch = self.torch
        tok, modelo = self.cargar(nombre)
        perfil = PERFILES.get(nombre, {"sistema": True, "muestreo": {}})
        if not perfil.get("sistema"):  # sin mensaje de sistema: va adelante del pedido
            sis = "\n\n".join(m["content"] for m in mensajes if m["role"] == "system")
            mensajes = [m for m in mensajes if m["role"] != "system"]
            if sis and mensajes:
                mensajes[0] = {"role": mensajes[0]["role"], "content": sis + "\n\n" + mensajes[0]["content"]}
        extra = {"enable_thinking": False} if perfil.get("sin_pensar") else {}
        try:
            ids = tok.apply_chat_template(mensajes, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, **extra)
        except TypeError:
            ids = tok.apply_chat_template(mensajes, add_generation_prompt=True, return_tensors="pt", **extra)
        if not isinstance(ids, dict) and not hasattr(ids, "keys"):
            ids = {"input_ids": ids}
        ids = {k: v.to(modelo.device) for k, v in ids.items()}
        if "attention_mask" not in ids:
            ids["attention_mask"] = torch.ones_like(ids["input_ids"])
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        gen = dict(max_new_tokens=int(max_new_tokens), pad_token_id=pad)
        muestreo = dict(perfil.get("muestreo", {}))
        temperatura = float(temperatura or 0)
        if temperatura > 0 or perfil.get("muestrear"):  # Hy-MT2 siempre con su muestreo recomendado
            if temperatura > 0 and not perfil.get("muestrear"):
                muestreo["temperature"] = temperatura
            gen.update(do_sample=True, **muestreo)
            torch.manual_seed(int(semilla if semilla is not None else 1234))
        else:
            gen.update(do_sample=False, repetition_penalty=max(1.05, muestreo.get("repetition_penalty", 1.0)))
        with torch.no_grad():
            out = modelo.generate(**ids, **gen)
        texto = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in texto:  # por si igual pensó en voz alta
            texto = texto.split("</think>", 1)[1]
        return texto.strip()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--probar":
        import torch
        import transformers
        log(f"transformers {transformers.__version__}, torch {torch.__version__}, "
            f"CUDA {'sí' if torch.cuda.is_available() else 'no'}")
        try:
            import bitsandbytes
            log(f"bitsandbytes {bitsandbytes.__version__}")
        except Exception as e:
            log(f"bitsandbytes no anda: {e}")
            return 1
        log("traductores listos")
        return 0
    modelos = None
    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        try:
            req = json.loads(linea)
            if req.get("op") == "fin":
                break
            if modelos is None:
                modelos = Modelos()
            if req.get("op") == "cargar":
                modelos.cargar(req["modelo"])
                resp = {"ok": True}
            else:
                resp = {"ok": True, "texto": modelos.chat(req["modelo"], req["mensajes"], req.get("max_new_tokens", 400),
                                                          req.get("temperatura", 0.0), req.get("semilla"))}
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        PROTO.write(json.dumps(resp, ensure_ascii=False) + "\n")
        PROTO.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
