"""Conversion de voz cantada (SVC) con YingMusic-SVC (GiantAILab, licencia MIT; basado en Seed-VC).

Cambia el timbre de una voz cantada sin tocar lo que canta: las notas, el ritmo, el vibrato y la expresion
salen de la voz de entrada; de la muestra solo se toma "como suena" la voz. Lo usa canto_motor.py cuando
el trabajo trae "tipo": "convertir":

    {"tipo": "convertir", "repo_svc": ".../YingMusic-SVC", "pasos": 40, "cfg_rate": 0.7,
     "cap": "cap01", "pct": [30, 45],
     "items": [{"id": "base_0", "fuente": "a.wav", "timbre": "muestra.wav", "salida": "b.wav",
                "guia": "melodia.wav" (opcional), "semitonos": null | 0 | 12 | -12 | "base_0"}]}

- fuente: lo que se canta (letra y ritmo).
- guia: si viene, las notas salen de aca (la melodia original), no de la fuente: asi un canto generado
  "cae" exactamente en las notas del original.
- timbre: muestra de la voz que tiene que sonar (hasta 25 s).
- semitonos: null = automatico (solo octavas, para llevar la melodia al registro de la voz destino);
  un numero = fijo; el id de otro item = el mismo corrimiento que se uso en ese (todo el tramo igual).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")  # el modelo importa matplotlib; sin ventanas

AQUI = Path(__file__).resolve().parent
SR = 44100
HOP = 512
CKPT = ("GiantAILab/YingMusic-SVC", "YingMusic-SVC-full.pt")
RMVPE = ("lj1995/VoiceConversionWebUI", "rmvpe.pt")
CAMPPLUS = ("funasr/campplus", "campplus_cn_common.bin")
VOCODER = "nvidia/bigvgan_v2_44khz_128band_512x"
WHISPER = "openai/whisper-small"


def bajar_pesos():
    """Deja en la cache de Hugging Face todo lo que usa la conversion (~2.5 GB). Se puede cortar y seguir."""
    from huggingface_hub import hf_hub_download, snapshot_download

    for repo, archivo in (CKPT, RMVPE, CAMPPLUS):
        print(f"[canto] bajando {repo}/{archivo} ...", flush=True)
        hf_hub_download(repo, archivo)
    print(f"[canto] bajando {VOCODER} ...", flush=True)
    snapshot_download(VOCODER, allow_patterns=["config.json", "bigvgan_generator.pt"])
    print(f"[canto] bajando {WHISPER} ...", flush=True)
    snapshot_download(WHISPER, allow_patterns=["*.json", "*.txt", "model.safetensors"])
    print("[canto] pesos de la conversion listos", flush=True)


def _cargar_estado(ruta):
    import torch

    try:
        return torch.load(ruta, map_location="cpu", weights_only=True)
    except Exception:  # checkpoints viejos con objetos de python adentro (repo fijo y conocido)
        return torch.load(ruta, map_location="cpu", weights_only=False)


class Conversor:
    def __init__(self, repo, dev=None, sin_pesos=False):
        import torch
        import yaml

        repo = Path(repo).resolve()
        os.chdir(repo)
        for p in (str(AQUI / "dac_min"), str(repo)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from modules.audio import mel_spectrogram
        from modules.campplus.DTDNN import CAMPPlus
        from modules.commons import build_model, recursive_munch
        from modules.rmvpe import RMVPE as Rmvpe

        self.torch = torch
        self.dev = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = self.dev.type == "cuda"
        cfg = yaml.safe_load(open(repo / "configs" / "YingMusic-SVC.yml", "r", encoding="utf-8"))
        mp = recursive_munch(cfg["model_params"])
        mp.dit_type = "DiT"
        self.estilo_residual = bool(cfg["model_params"]["length_regulator"].get("use_style_residual", False))
        modelo = build_model(mp, stage="DiT")
        if not sin_pesos:
            from huggingface_hub import hf_hub_download

            params = _cargar_estado(hf_hub_download(*CKPT))["net"]
            for k in modelo:
                if k not in params:
                    continue
                sd = {(n[7:] if n.startswith("module.") else n): v for n, v in params[k].items()}
                propio = modelo[k].state_dict()
                ok = {n: v for n, v in sd.items() if n in propio and v.shape == propio[n].shape}
                if len(ok) < len(propio):
                    print(f"[canto] aviso: {k}: {len(propio) - len(ok)} parametros sin cargar", flush=True)
                modelo[k].load_state_dict(ok, strict=False)
        for k in modelo:
            modelo[k].eval().to(self.dev)
        modelo.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.modelo = modelo

        if sin_pesos:  # solo para probar el circuito sin bajar nada
            import tempfile
            ruta = Path(tempfile.gettempdir()) / "_rmvpe_prueba.pt"
            if not ruta.exists():
                from modules.rmvpe import E2E
                torch.save(E2E(4, 1, (2, 2)).state_dict(), ruta)
        else:
            from huggingface_hub import hf_hub_download
            ruta = hf_hub_download(*RMVPE)
        self.rmvpe = Rmvpe(str(ruta), is_half=False, device=self.dev)

        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        if not sin_pesos:
            from huggingface_hub import hf_hub_download
            self.campplus.load_state_dict(_cargar_estado(hf_hub_download(*CAMPPLUS)))
        self.campplus.eval().to(self.dev)

        from modules.bigvgan import bigvgan
        if sin_pesos:
            from modules.bigvgan.env import AttrDict
            h = AttrDict(dict(num_mels=128, sampling_rate=SR, hop_size=HOP, n_fft=2048, win_size=2048, fmin=0,
                              fmax=None, resblock="1", upsample_rates=[8, 4, 2, 2, 2, 2],
                              upsample_kernel_sizes=[16, 8, 4, 4, 4, 4], upsample_initial_channel=64,
                              resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3, 5]],
                              activation="snakebeta", snake_logscale=True, use_tanh_at_final=False,
                              use_bias_at_final=False))
            voc = bigvgan.BigVGAN(h, use_cuda_kernel=False)
        else:
            voc = bigvgan.BigVGAN.from_pretrained(VOCODER, use_cuda_kernel=False)
        voc.remove_weight_norm()
        self.vocoder = voc.eval().to(self.dev)

        from transformers import AutoFeatureExtractor, WhisperConfig, WhisperFeatureExtractor, WhisperModel
        if sin_pesos:
            wm = WhisperModel(WhisperConfig(d_model=768, encoder_layers=2, decoder_layers=1,
                                            encoder_attention_heads=12, decoder_attention_heads=12,
                                            encoder_ffn_dim=1024, decoder_ffn_dim=1024, num_mel_bins=80))
            self.fe = WhisperFeatureExtractor(feature_size=80)
        else:
            wm = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float16 if self.fp16 else torch.float32)
            self.fe = AutoFeatureExtractor.from_pretrained(WHISPER)
        del wm.decoder
        self.whisper = wm.eval().to(self.dev)

        sp = cfg["preprocess_params"]["spect_params"]
        self.mel_args = {"n_fft": sp["n_fft"], "win_size": sp["win_length"], "hop_size": sp["hop_length"],
                         "num_mels": sp["n_mels"], "sampling_rate": SR, "fmin": sp.get("fmin", 0),
                         "fmax": None, "center": False}
        self._mel = mel_spectrogram

    # ------------------------------------------------------------ piezas
    def mel(self, x):
        return self._mel(x, **self.mel_args)

    def semantica(self, w16):
        """Contenido (lo que se dice/canta) con el encoder de Whisper; en ventanas de 30 s con solape de 5 s."""
        torch = self.torch

        def una(trozo):
            x = self.fe([trozo.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True,
                        sampling_rate=16000)
            feats = self.whisper._mask_input_features(x.input_features, attention_mask=x.attention_mask)
            with torch.no_grad():
                out = self.whisper.encoder(feats.to(self.dev).to(self.whisper.encoder.dtype), head_mask=None,
                                           output_attentions=False, output_hidden_states=False, return_dict=True)
            s = out.last_hidden_state.to(torch.float32)
            return s[:, :trozo.size(-1) // 320 + 1]

        if w16.size(-1) <= 16000 * 30:
            return una(w16)
        solape, partes, buf, hecho = 5, [], None, 0
        while hecho < w16.size(-1):
            if buf is None:
                trozo = w16[:, hecho:hecho + 16000 * 30]
            else:
                trozo = torch.cat([buf, w16[:, hecho:hecho + 16000 * (30 - solape)]], dim=-1)
            s = una(trozo)
            partes.append(s if hecho == 0 else s[:, 50 * solape:])
            buf = trozo[:, -16000 * solape:]
            hecho += 30 * 16000 if hecho == 0 else trozo.size(-1) - 16000 * solape
        return torch.cat(partes, dim=1)

    def f0(self, w16):
        return self.rmvpe.infer_from_audio(w16[0], thred=0.03)

    @staticmethod
    def guiar(f0_fuente, f0_guia, hueco_max=25):
        """Las notas de la guia donde la fuente tiene voz; los huecos cortos de la guia se rellenan
        (consonantes), y donde la guia no tiene nota se deja la de la fuente llevada a la octava mas cercana."""
        n = len(f0_fuente)
        g = np.zeros(n, np.float64)
        m = min(n, len(f0_guia))
        g[:m] = f0_guia[:m]
        v = g > 1
        if v.sum() >= 2:
            idx = np.arange(n)
            lg = np.interp(idx, idx[v], np.log(g[v]))
            # solo huecos cortos (y no antes/despues de la primera/ultima nota)
            relleno = np.zeros(n, bool)
            i = 0
            while i < n:
                if not v[i]:
                    j = i
                    while j < n and not v[j]:
                        j += 1
                    if i > 0 and j < n and j - i <= hueco_max:
                        relleno[i:j] = True
                    i = j
                else:
                    i += 1
            g = np.where(v, g, np.where(relleno, np.exp(lg), 0.0))
        out = np.zeros(n, np.float64)
        voz = f0_fuente > 1
        con_guia = voz & (g > 1)
        out[con_guia] = g[con_guia]
        solo = voz & ~(g > 1)
        if solo.any() and (g > 1).any():
            ref = np.median(g[g > 1])
            k = np.round(np.log2(ref / f0_fuente[solo]))
            out[solo] = f0_fuente[solo] * (2.0 ** k)
        elif solo.any():
            out[solo] = f0_fuente[solo]
        return out, float(con_guia.sum()) / max(1, int(voz.sum()))

    @staticmethod
    def corrimiento(f0_timbre, f0_canto):
        """Corrimiento automatico en octavas (0 o +-12), como el original de YingMusic-SVC."""
        a, b = f0_timbre[f0_timbre > 1], f0_canto[f0_canto > 1]
        if len(a) == 0 or len(b) == 0:
            return 0
        base = 12 * (np.mean(np.log(a)) - np.mean(np.log(b))) / np.log(2)
        media = float(np.mean(a))
        if media > 205:
            factor = max(0.3, 1.0 - (media - 205) / (560 - 205))
        elif media < 120:
            factor = 1.0 - (120 - media) / 120 * 1.2
        else:
            factor = 1.0
        s = float(np.clip(base * factor + 3.5, -24, 24))
        return 12 if s >= 7 else (-12 if s <= -7 else 0)

    # ------------------------------------------------------------ conversion
    def convertir(self, fuente, timbre, guia=None, semitonos=None, pasos=40, cfg_rate=0.7, semilla=1234):
        import librosa
        import torchaudio

        torch = self.torch
        torch.manual_seed(int(semilla))
        x = librosa.load(str(fuente), sr=SR)[0]
        r = librosa.load(str(timbre), sr=SR)[0][:SR * 25]
        dur = len(x)
        with torch.no_grad():
            src = torch.tensor(x).unsqueeze(0).float().to(self.dev)
            ref = torch.tensor(r).unsqueeze(0).float().to(self.dev)
            src16 = torchaudio.functional.resample(src, SR, 16000)
            ref16 = torchaudio.functional.resample(ref, SR, 16000)
            s_alt = self.semantica(src16)
            s_ori = self.semantica(ref16)
            mel = self.mel(src.float())
            mel2 = self.mel(ref.float())
            largo = torch.LongTensor([mel.size(2)]).to(self.dev)
            largo2 = torch.LongTensor([mel2.size(2)]).to(self.dev)
            fb = torchaudio.compliance.kaldi.fbank(ref16, num_mel_bins=80, dither=0, sample_frequency=16000)
            fb = fb - fb.mean(dim=0, keepdim=True)
            estilo = self.campplus(fb.unsqueeze(0))

            f0_ref = self.f0(ref16)
            f0_src = self.f0(src16)
            cobertura = None
            if guia is not None:
                g = librosa.load(str(guia), sr=16000)[0]
                f0_g = self.f0(torch.tensor(g).unsqueeze(0).float().to(self.dev))
                f0_obj, cobertura = self.guiar(f0_src, f0_g)
            else:
                f0_obj = f0_src.astype(np.float64)
            if semitonos is None:
                semitonos = self.corrimiento(f0_ref, f0_obj)
            f0_obj = np.where(f0_obj > 1, f0_obj * 2.0 ** (float(semitonos) / 12.0), 0.0)
            F0_ref = torch.from_numpy(np.asarray(f0_ref, np.float32)).to(self.dev)[None]
            F0_obj = torch.from_numpy(np.asarray(f0_obj, np.float32)).to(self.dev)[None]

            cond, _, _, _, _, estilo_c = self.modelo.length_regulator(
                s_alt, ylens=largo, n_quantizers=3, f0=F0_obj, style=estilo, return_style_residual=True)
            prompt, _, _, _, _, estilo_p = self.modelo.length_regulator(
                s_ori, ylens=largo2, n_quantizers=3, f0=F0_ref, style=estilo, return_style_residual=True)

            ventana = SR // HOP * 30 - mel2.size(2)
            solape = 16
            solape_w = solape * HOP
            hecho, partes, previo = 0, [], None
            while hecho < cond.size(1):
                c = cond[:, hecho:hecho + ventana]
                ultimo = hecho + ventana >= cond.size(1)
                cc = torch.cat([prompt, c], dim=1)
                ec = None
                if self.estilo_residual:
                    ec = torch.cat([estilo_p, estilo_c[:, hecho:hecho + ventana]], dim=1)
                with torch.autocast(device_type=self.dev.type, dtype=torch.float16 if self.fp16 else torch.float32,
                                    enabled=self.fp16):
                    m = self.modelo.cfm.inference(cc, torch.LongTensor([cc.size(1)]).to(self.dev), mel2, estilo,
                                                  None, int(pasos), inference_cfg_rate=float(cfg_rate), style_r=ec)
                    m = m[:, :, mel2.size(-1):]
                w = self.vocoder(m.float()).squeeze()[None, :][0].float().cpu().numpy()
                if previo is None:
                    if ultimo:
                        partes.append(w)
                        break
                    partes.append(w[:-solape_w])
                else:
                    w = w.copy()
                    k = min(solape_w, len(w))
                    fo = np.cos(np.linspace(0, np.pi / 2, solape_w)) ** 2
                    fi = np.cos(np.linspace(np.pi / 2, 0, solape_w)) ** 2
                    w[:k] = w[:k] * fi[:k] + (previo * fo)[:k]
                    if ultimo:
                        partes.append(w)
                        break
                    partes.append(w[:-solape_w])
                previo = w[-solape_w:]
                hecho += m.size(2) - solape
        y = np.concatenate(partes).astype(np.float32)
        y = np.pad(y, (0, max(0, dur - len(y))))[:dur]
        return y, int(semitonos), cobertura


def correr(trabajo, progreso):
    """Hace todos los items de un trabajo "convertir"; devuelve la lista de resultados."""
    import soundfile as sf

    items = trabajo["items"]
    progreso(trabajo, 0.0, "cargando el conversor de voz")
    print("[canto] cargando YingMusic-SVC (la primera vez baja ~2.5 GB)...", flush=True)
    conv = Conversor(trabajo["repo_svc"], sin_pesos=bool(os.environ.get("CANTO_SIN_PESOS")))
    print(f"[canto] conversor listo en {conv.dev}", flush=True)
    usados, resultados = {}, []
    for k, it in enumerate(items):
        t0 = time.time()
        progreso(trabajo, (k + 0.05) / max(1, len(items)), f"llevando la voz: {k + 1} de {len(items)}")
        r = {"id": it["id"], "ok": False}
        try:
            st = it.get("semitonos")
            if isinstance(st, str):
                st = usados.get(st)
            y, st, cob = conv.convertir(it["fuente"], it["timbre"], it.get("guia"), st,
                                        trabajo.get("pasos", 40), trabajo.get("cfg_rate", 0.7),
                                        it.get("semilla", 1234))
            Path(it["salida"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(it["salida"], y, SR, subtype="FLOAT")
            usados[it["id"]] = st
            r.update(ok=True, semitonos=st, cobertura=None if cob is None else round(cob, 3),
                     dur=round(len(y) / SR, 3))
        except Exception as e:
            import traceback
            r["error"] = f"{type(e).__name__}: {e}"
            print(f"[canto] ERROR convirtiendo {it['id']}:\n{traceback.format_exc()}", flush=True)
        r["seg"] = round(time.time() - t0, 1)
        if r["ok"]:
            extra = f", notas de la guía en {100 * r['cobertura']:.0f}%" if r.get("cobertura") is not None else ""
            print(f"[canto] {it['id']}: {r['dur']:.1f} s convertidos en {r['seg']:.0f} s "
                  f"(corrimiento {st:+d} semitonos{extra})", flush=True)
        resultados.append(r)
        yield resultados
    progreso(trabajo, 1.0, "voz lista")
