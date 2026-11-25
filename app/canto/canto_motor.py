"""Motor de canto: vuelve a cantar un tramo con otra letra, conservando la melodia original.

Corre en su propio entorno (app/env_canto) con YingMusic-Singer-Plus (ASLP-lab, CC BY 4.0; el VAE es de
Stable Audio Open, Stability AI Community License). Lo llama paso3_doblar.py asi:

    env_canto/python canto/canto_motor.py <trabajo.json>

trabajo.json:
    {"repo": ".../YingMusic-Singer-Plus", "modelo": "ASLP-lab/YingMusic-Singer-Plus", "nfe": 32, "cfg": 3.0,
     "cap": "cap01", "pct": [2, 30],
     "items": [{"id": "t0_k0", "melodia": "x.wav", "ref": "r.wav", "ref_texto": "...", "ref_idioma": "es",
                "texto": "...", "idioma": "en", "salida": "y.wav", "semilla": 1234}]}

Deja <trabajo>_resultado.json con {"items": [{"id", "ok", "error", "dur"}]} y va imprimiendo @@P {...}.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

AQUI = Path(__file__).resolve().parent

# espeak-ng sin instalar nada en Windows: la libreria viene dentro del paquete espeakng-loader
try:
    import espeakng_loader

    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
except Exception as e:  # en Linux/Mac puede estar el espeak del sistema
    print(f"[canto] aviso: espeakng-loader no disponible ({e})", flush=True)

# lo que espeak usa para cada idioma; en/de/fr van por el camino original del modelo
ESPEAK = {"es": "es-419", "ru": "ru", "it": "it", "pt": "pt", "en": "en-us", "de": "de", "fr": "fr-fr"}
PROPIOS = {"en", "de", "fr"}  # el tokenizador de YingMusic ya los conoce

# fonemas que el vocabulario del modelo no tiene -> el mas parecido que si tiene
MAPA = {
    "ɣ": "ɡ", "β": "b", "ʎ": "j", "ʝ": "j", "ɟ": "ɡ", "c": "k", "ɨ": "ɪ", "ʂ": "ʃ", "ʐ": "ʒ", "ɵ": "o",
    "ʉ": "u", "ɭ": "l", "ʏ": "y", "ɔø": "ɔɪ", "oɪ": "ɔɪ", "eɪ̯": "eɪ", "ɐ̃": "ɐ", "ɛ̃": "ɛ", "ẽ": "e",
    "õ": "o", "ĩ": "i", "ũ": "u", "ɾʲ": "ɾ", "ʃʲ": "ʃ", "tʃʲ": "tʃ", "ʃː": "ʃ", "ɕː": "ɕ", "ʑː": "ʑ",
    "ɪ̯": "ɪ", "ʊ̯": "ʊ", "ʔ": "", "ʰ": "", "ˈ": "", "ˌ": "", "-": "",
}


def a_tokens(fonos, vocab, cod):
    """Lista de fonemas de espeak -> fonemas del vocabulario del modelo (lo que no se puede, se descarta)."""
    out = []
    for p in fonos:
        p = p.strip()
        if not p:
            continue
        if cod == "ru" and p == "y":  # en ruso espeak escribe la 'ы' como y
            p = "ɪ"
        if p in vocab:
            out.append(p)
            continue
        q = MAPA.get(p)
        if q is not None:
            if q in vocab:
                out.append(q)
            continue
        q = p.replace("ː", "")
        if q in vocab:
            out.append(q)
            continue
        if p.endswith("ʲ") and p[:-1] in vocab:  # palatalizadas del ruso: base + ʲ
            out += [p[:-1], "ʲ"]
            continue
        # ultimo recurso: caracter por caracter
        for ch in p:
            ch = MAPA.get(ch, ch)
            if ch and ch in vocab:
                out.append(ch)
    return out


class Tokenizador:
    """Reemplaza al tokenizador chino/ingles del modelo para cantar en mas idiomas (fonemas IPA de espeak).
    'cola' dice en que idioma esta cada linea que va a pedir el modelo (primero la referencia, despues la letra)."""

    def __init__(self, original):
        from src.YingMusicSinger.utils.f5_tts.g2p.g2p.text_tokenizers import TextTokenizer
        from src.YingMusicSinger.utils.f5_tts.g2p.g2p_generation import text_tokenizer, vocab

        self.original = original
        self.phone2id = original.phone2id
        self.vocab = vocab
        self.propio = text_tokenizer
        self.TextTokenizer = TextTokenizer
        self.espeak = {}
        self.cola = []

    def fonemas(self, texto, cod):
        cod = cod if cod in ESPEAK else "es"
        if cod in PROPIOS:
            ph, tok = self.propio.tokenize(texto, texto, cod)
            return [p for p in ph.split("|") if p], [t for t in tok]
        if cod not in self.espeak:
            self.espeak[cod] = self.TextTokenizer(language=ESPEAK[cod])
        ph = self.espeak[cod](texto)
        fonos = a_tokens(ph.split("|"), self.vocab, cod)
        return fonos, [self.vocab[p] for p in fonos]

    def encode(self, texto):
        cod = self.cola.pop(0) if self.cola else "en"
        _, tok = self.fonemas(texto, cod)
        return [t + 1 for t in tok]

    def decode(self, tokens):
        return self.original.decode(tokens)


def progreso(trabajo, frac, det):
    a, b = trabajo.get("pct", [0, 100])
    print("@@P " + json.dumps({"cap": trabajo.get("cap", ""), "paso": 3, "pct": round(a + (b - a) * frac, 1),
                               "det": det}, ensure_ascii=False), flush=True)


def main():
    ruta = Path(sys.argv[1]).resolve()
    trabajo = json.loads(ruta.read_text(encoding="utf-8"))
    repo = Path(trabajo["repo"]).resolve()
    os.chdir(repo)  # el modelo abre sus archivos con rutas relativas a su carpeta
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(AQUI / "dac_min"))
    import torch
    import torchaudio

    items = trabajo["items"]
    res_ruta = ruta.with_name(ruta.stem + "_resultado.json")
    resultados = []
    progreso(trabajo, 0.0, "cargando el modelo de canto")
    print("[canto] cargando YingMusic-Singer-Plus (la primera vez descarga ~3 GB)...", flush=True)
    from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("CANTO_SIN_PESOS"):  # solo para probar el circuito sin bajar el modelo
        modelo = YingMusicSinger(model_cfg_path="./src/YingMusicSinger/config/YingMusic_Singer.yaml",
                                 vae_config_path="./src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    else:
        modelo = YingMusicSinger.from_pretrained(trabajo.get("modelo", "ASLP-lab/YingMusic-Singer-Plus"))
    modelo = modelo.to(dev).eval()
    tok = Tokenizador(modelo.cnen_tokenizer)
    modelo.cnen_tokenizer = tok
    print(f"[canto] modelo listo en {dev}", flush=True)

    for k, it in enumerate(items):
        t0 = time.time()
        progreso(trabajo, (k + 0.05) / max(1, len(items)), f"cantando: trozo {k + 1} de {len(items)}")
        r = {"id": it["id"], "ok": False}
        try:
            tok.cola = [it.get("ref_idioma", "es"), it.get("idioma", "en")]
            audio, sr = modelo(
                ref_audio_path=it["ref"], melody_audio_path=it["melodia"],
                ref_text=it.get("ref_texto", "").strip() or ".", target_text=it["texto"].strip(),
                lrc_align_mode="sentence_level", sil_len_to_end=0.5, t_shift=0.5,
                nfe_step=int(trabajo.get("nfe", 32)), cfg_strength=float(trabajo.get("cfg", 3.0)),
                seed=int(it.get("semilla", 666)),
            )
            Path(it["salida"]).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(it["salida"], audio.to("cpu"), sample_rate=sr)
            r.update(ok=True, dur=round(audio.shape[-1] / sr, 3), seg=round(time.time() - t0, 1))
            print(f"[canto] {it['id']}: {r['dur']:.1f} s en {r['seg']:.0f} s", flush=True)
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            print(f"[canto] ERROR en {it['id']}:\n{traceback.format_exc()}", flush=True)
        resultados.append(r)
        res_ruta.write_text(json.dumps({"items": resultados}, ensure_ascii=False, indent=1), encoding="utf-8")
    progreso(trabajo, 1.0, "canto listo")


def probar_fonemas():
    """python canto_motor.py --fonemas <repo>: muestra como quedan los fonemas en cada idioma."""
    repo = Path(sys.argv[2]).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

    t = Tokenizador(CNENTokenizer())
    for cod, txt in [("es", "Hoy quiero cantar contigo, corazón, mañana llueve"),
                     ("en", "Tonight I want to sing with you, my heart"),
                     ("de", "Ich möchte heute mit dir singen, mein Herz"),
                     ("ru", "Сегодня я хочу петь с тобой, моё сердце"),
                     ("it", "Voglio cantare con te"), ("pt", "Quero cantar com você")]:
        fonos, tok = t.fonemas(txt, cod)
        print(cod, len(tok), " ".join(fonos))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--fonemas":
        probar_fonemas()
    else:
        main()
