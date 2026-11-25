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

La primera vez se descargan los modelos (Whisper, Demucs, Qwen3-TTS y el traductor local, unos 16 GB).
En `_proceso/capNN/revisar_XX.txt` queda una lista de lo dudoso de cada video, para revisarlo.

Importante: cloná solo voces de personas que te hayan dado permiso.
