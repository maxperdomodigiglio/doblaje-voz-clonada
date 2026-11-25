# Doblaje automático con voz clonada (app de Pinokio)

Traduce y dobla **videos o audios** al **inglés, alemán, ruso o castellano**, con la **voz clonada** del que habla
(o con una muestra de voz que subas), conserva la música y el ambiente original, y agrega **subtítulos**.
El idioma original se detecta solo (castellano, inglés, alemán, ruso, francés, italiano, portugués…).
Todo corre en tu compu: **no necesita API key ni cuentas**.

## Qué necesitás
- Windows con placa de video **NVIDIA** (ideal 12 GB de memoria de video o más).
- **Pinokio** instalado (https://pinokio.co).
- Unos **30 GB libres** en disco (programas + modelos que se descargan la primera vez).

## Instalación (una sola vez)
**Desde GitHub (recomendado):** en Pinokio elegí descargar desde un link (*Download from URL*), pegá el link
de este repositorio y después tocá **Instalar** en la app. Tarda un rato.

**Desde un zip:** descomprimí la carpeta `doblaje-gregorio` dentro de la carpeta `api` de Pinokio
(normalmente `C:\pinokio\api\doblaje-gregorio`), abrí Pinokio (o refrescá la página), entrá a la app
**Doblaje Gregorio** y tocá **Instalar**.

## Uso
1. Tocá **Iniciar** y después **Abrir interfaz**.
2. **📁 Elegir…** la carpeta donde están los videos o audios (.mp4, .mov, .mkv, .mp3, .wav, .m4a…).
3. Elegí el idioma del doblaje (Inglés / Alemán / Ruso / Español) y la voz:
   - *Voz original de cada video*: clona la voz del que habla en cada video.
   - *Muestra de voz propia*: subí 10-30 s de una sola persona hablando, sin música.
4. **▶ Hacer todo**. Los resultados quedan en la subcarpeta `salida`:
   - Videos: `... [EN].mp4` (subtítulos seleccionables), `... [EN] subs ES.mp4` (subtítulos grabados) y los `.srt`.
   - Audios: `... [EN].mp3` con la voz doblada, más los `.srt` del original y de la traducción.

Los subtítulos van en castellano cuando se puede (si el original es castellano, o si se dobla al castellano);
si no, en el idioma original. Si el audio ya está en el idioma elegido (por ejemplo castellano → castellano),
no se traduce: solo se cambia la voz.

En **⚙️ Opciones avanzadas** podés fijar el idioma original (si la detección automática se equivoca) y contar
de qué trata el material (quién habla y sobre qué): ayuda a traducir mejor y queda guardado para esa carpeta.

Mientras trabaja, cada video muestra su barra de avance y abajo el tiempo que falta.
- **⏸ Pausar** frena todo y libera la placa de video y el procesador; **▶ Continuar** sigue en el mismo punto.
- **⏹ Detener** corta y libera también la memoria; después, **Hacer todo** sigue desde donde quedó.

No pongas la muestra de voz dentro de la carpeta de los videos: se doblaría como un audio más.

## 🎵 Canciones (opcional)
Si un video o audio tiene partes cantadas, la app las detecta sola y las **vuelve a cantar en el idioma del
doblaje, con la misma melodía y el timbre del cantante** (YingMusic-Singer-Plus). Lo hablado se dobla como siempre.

1. En Pinokio tocá **🎵 Instalar canto** (una sola vez, ~8 GB más; el modelo de canto, ~3 GB, se baja la primera
   vez que canta).
2. En la interfaz dejá tildado **🎵 Cantar las canciones** y tocá **▶ Hacer todo**.

- La letra se traduce buscando la misma cantidad de sílabas por línea, para que entre en las notas.
- Las partes cantadas van a velocidad normal (el video no se acelera ahí) y la música no se baja.
- Si una canción ya está en el idioma del doblaje, queda la original.
- El inglés es el que mejor canta; alemán, ruso y castellano salen con pronunciación aproximada (experimental).
- Si algún trozo no se puede cantar, queda la voz original en ese trozo.
- En cada video aparece 🎵 con la cantidad de partes cantadas que encontró.

La primera vez se descargan los modelos (Whisper, Demucs, Qwen3-TTS y el traductor local, unos 16 GB).
En `_proceso/capNN/revisar_XX.txt` queda una lista de lo dudoso de cada video, para revisarlo.

Importante: cloná solo voces de personas que te hayan dado permiso.

## Créditos
Demucs (Meta), Whisper (OpenAI), Qwen3-TTS y Qwen3 (Alibaba), YingMusic-Singer-Plus (ASLP-lab, CC BY 4.0; su VAE
es de Stable Audio Open, Stability AI Community License), detector de sonidos AST (MIT), espeak-ng, y las capas
de descript-audio-codec (MIT) en `app/canto/dac_min`.
