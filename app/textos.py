"""Herramientas de texto compartidas: correcciones con tiempos por palabra, subtitulos y bloques de doblaje."""
from comun import partir_lineas

FIN = (".", "?", "!", "…", ".»", "?»", "!»", ".)")


def corregir_segmentos(segmentos, correcciones):
    """correcciones: {inicio_aprox_seg: texto_corregido}. Reasigna tiempos por palabra."""
    out = []
    for sg in segmentos:
        txt = sg["text"]
        borrar = False
        for t0, nuevo in correcciones.items():
            if abs(sg["start"] - t0) < 0.15:
                if nuevo is None:
                    borrar = True
                else:
                    txt = nuevo
        if borrar:
            continue
        palabras_nuevas = txt.split()
        ws = sg["words"]
        if len(palabras_nuevas) == len(ws):
            words = [{"w": n, "s": w["s"], "e": w["e"]} for n, w in zip(palabras_nuevas, ws) if n != "∅"]
            txt = " ".join(w["w"] for w in words)
        else:  # repartir proporcional a la cantidad de letras
            total = sum(len(n) for n in palabras_nuevas) or 1
            t, dur = sg["start"], sg["end"] - sg["start"]
            words = []
            for n in palabras_nuevas:
                d = dur * len(n) / total
                words.append({"w": n, "s": t, "e": t + d})
                t += d
        out.append({"start": sg["start"], "end": sg["end"], "text": txt, "words": words})
    return out


def lineas_balanceadas(texto, max_chars=42):
    if len(texto) <= max_chars:
        return texto
    pal = texto.split()
    mejor, mejor_d = None, 1e9
    for i in range(1, len(pal)):
        a, b = " ".join(pal[:i]), " ".join(pal[i:])
        if len(a) <= max_chars and len(b) <= max_chars:
            d = abs(len(a) - len(b))
            if d < mejor_d:
                mejor, mejor_d = a + "\n" + b, d
    return mejor or "\n".join(partir_lineas(texto, max_chars)[:2])


def _txt(ws):
    return " ".join(w["w"] for w in ws)


def _cabe(t, ancho=42):
    if len(t) <= ancho:
        return True
    pal = t.split()
    return any(len(" ".join(pal[:i])) <= ancho and len(" ".join(pal[i:])) <= ancho for i in range(1, len(pal)))


def _partir(ws, max_chars=84, max_dur=8.0):
    t = _txt(ws)
    if len(ws) < 2 or (_cabe(t) and ws[-1]["e"] - ws[0]["s"] <= max_dur):
        return [ws]
    mejor, mejor_c = 1, 1e9
    for i in range(1, len(ws)):
        a, b = _txt(ws[:i]), _txt(ws[i:])
        c = abs(len(a) - len(b)) / max(len(t), 1)
        if not ws[i - 1]["w"].endswith((",", ";", ":", ".", "?", "!", "…")):
            c += 0.35
        if ws[i - 1]["w"].endswith((".", "?", "!")):
            c -= 0.1
        if len(ws[i:]) == 1 or i == 1:
            c += 0.5  # evitar palabras huerfanas
        if any(ch.isdigit() for ch in ws[i - 1]["w"]) and any(ch.isdigit() for ch in ws[i]["w"]):
            c += 0.6  # no separar "10, 15" 
        if c < mejor_c:
            mejor, mejor_c = i, c
    return _partir(ws[:mejor], max_chars, max_dur) + _partir(ws[mejor:], max_chars, max_dur)


def cues_desde_palabras(segmentos, max_chars=84, max_dur=8.0, min_dur=1.0):
    # 0) todas las palabras en orden, cortadas en oraciones (sin importar los cortes de Whisper)
    ws_all = [w for sg in segmentos for w in sg["words"]]
    frases, cur = [], []
    for k, w in enumerate(ws_all):
        cur.append(w)
        sig = ws_all[k + 1] if k + 1 < len(ws_all) else None
        if (w["w"].endswith((".", "?", "!", "…", ".»", "?»", "!»", ".)")) or sig is None
                or sig["s"] - w["e"] > 1.5):
            frases.append(cur)
            cur = []
    # 1) unir frases cortadas por una pausa (no terminadas en punto) si entran juntas
    grupos = []
    for ws in frases:
        if grupos:
            g = grupos[-1]
            gap = ws[0]["s"] - g[-1]["e"]
            junto = _txt(g + ws)
            if (gap < 0.8 and _cabe(junto) and ws[-1]["e"] - g[0]["s"] <= max_dur
                    and not g[-1]["w"].endswith((".", "?", "!", "…", ".»", "?»", "!»"))):
                g.extend(ws)
                continue
        grupos.append(list(ws))
    # 2) partir los largos en partes parejas, preferentemente en comas
    cues = []
    for g in grupos:
        for parte in _partir(g, max_chars, max_dur):
            cues.append([parte[0]["s"], parte[-1]["e"], _txt(parte)])
    # 3) duraciones legibles
    for i, c in enumerate(cues):
        sig = cues[i + 1][0] if i + 1 < len(cues) else c[1] + 3
        c[1] = min(max(c[1] + 0.3, c[0] + min_dur), sig - 0.05)
        c[2] = lineas_balanceadas(c[2])
    return cues


def palabras_limpias(es, excluir=()):
    ws = []
    for sg in es["segmentos"]:
        if sg["id"] in excluir:
            continue
        for w in sg["words"]:
            s, e = w["s"], w["e"]
            if e - s > 0.9:  # pausa escondida dentro de la palabra
                s = e - 0.5
            ws.append({"w": w["w"], "s": s, "e": e})
    return ws


def bloques_auto(ws, max_dur=13.0, min_dur=2.0):
    bloques, cur = [], []

    def cerrar():
        if cur:
            bloques.append([cur[0]["s"], cur[-1]["e"], " ".join(x["w"] for x in cur)])

    for w in ws:
        if cur:
            gap = w["s"] - cur[-1]["e"]
            dur = cur[-1]["e"] - cur[0]["s"]
            ult = cur[-1]["w"]
            fin = ult.endswith((".", "?", "!", "…"))
            coma = ult.endswith((",", ";", ":"))
            if (gap >= 1.2 or (gap >= 0.5 and (fin or coma) and dur >= min_dur) or (fin and dur >= 6)
                    or (dur >= max_dur and (fin or coma or gap >= 0.25)) or dur >= max_dur * 1.4):
                cerrar()
                cur = []
        cur.append(w)
    cerrar()
    return bloques


def _corte_mejor(ws):
    """Indice donde conviene partir una oracion larga: pausa larga y/o coma, cerca del medio."""
    n = len(ws)
    t0, t1 = ws[0]["s"], ws[-1]["e"]
    mejor, mejor_c = n // 2, -1e9
    for i in range(1, n):
        gap = ws[i]["s"] - ws[i - 1]["e"]
        medio = 1 - abs(((ws[i]["s"] - t0) / max(t1 - t0, 0.01)) - 0.5) * 2  # 1 en el medio, 0 en las puntas
        c = min(gap, 1.5) * 2 + medio * 1.5
        if ws[i - 1]["w"].endswith((",", ";", ":")):
            c += 1.5
        if i < 3 or n - i < 3:
            c -= 3  # nada de pedacitos de 1-2 palabras
        if c > mejor_c:
            mejor, mejor_c = i, c
    return mejor


def bloques_por_oraciones(ws, max_dur=13.0, max_gap=3.0, min_dur=4.0):
    """Bloques de doblaje que respetan las oraciones (mejor para traducir que cortar en cada pausa)."""
    oraciones, cur = [], []
    for k, w in enumerate(ws):
        if cur and w["s"] - cur[-1]["e"] > max_gap:
            oraciones.append(cur)
            cur = []
        cur.append(w)
        if w["w"].endswith(FIN):
            oraciones.append(cur)
            cur = []
    if cur:
        oraciones.append(cur)

    def partir(o):
        if o[-1]["e"] - o[0]["s"] <= max_dur * 1.15 or len(o) < 6:
            return [o]
        i = _corte_mejor(o)
        return partir(o[:i]) + partir(o[i:])

    piezas = [p for o in oraciones for p in partir(o)]
    bloques = []
    for p in piezas:
        if bloques:
            b = bloques[-1]
            gap = p[0]["s"] - b[-1]["e"]
            junto = p[-1]["e"] - b[0]["s"]
            corto = (b[-1]["e"] - b[0]["s"] < min_dur) or (p[-1]["e"] - p[0]["s"] < min_dur)
            if gap < 1.5 and (junto <= max_dur or (corto and junto <= max_dur * 1.25)):
                b.extend(p)
                continue
        bloques.append(list(p))
    return [[b[0]["s"], b[-1]["e"], " ".join(x["w"] for x in b)] for b in bloques]
