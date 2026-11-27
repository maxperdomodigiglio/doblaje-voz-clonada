# Traductor de videos (app de Pinokio)

Traduce y dobla **videos o audios** al **inglés, alemán, ruso o castellano**, con la **voz clonada** del que habla
(o con una muestra de voz que subas), conserva la música y el ambiente original, y agrega **subtítulos**.
El idioma original se detecta solo (castellano, inglés, alemán, ruso, francés, italiano, portugués…).
Todo corre en tu compu: **no necesita API key ni cuentas**. La interfaz se ve clara u oscura según cómo tengas
configurado Windows (o el navegador).

## Qué necesitás
- Windows con placa de video **NVIDIA** (ideal 12 GB de memoria de video o más).
- **Pinokio** instalado (https://pinokio.co).
- Unos **30 GB libres** en disco (programas + modelos que se descargan la primera vez).

## Instalación (una sola vez)
**Desde GitHub (recomendado):** en Pinokio elegí descargar desde un link (*Download from URL*), pegá el link
de este repositorio y después tocá **Instalar** en la app. Tarda un rato.

**Desde un zip:** descomprimí la carpeta `doblaje-voz-clonada` dentro de la carpeta `api` de Pinokio
(normalmente `C:\pinokio\api\doblaje-voz-clonada`), abrí Pinokio (o refrescá la página), entrá a la app
**Traductor de videos** y tocá **Instalar**.

## Uso
1. Tocá **Iniciar** y después **Abrir interfaz**.
2. **Elegir…** la carpeta donde están los videos o audios (.mp4, .mov, .mkv, .mp3, .wav, .m4a…).
3. Elegí el idioma del doblaje (Inglés / Alemán / Ruso / Español) y la voz:
   - *Voz original de cada video*: clona la voz del que habla en cada video.
   - *Muestra de voz propia*: subí 10-30 s de una sola persona hablando, sin música.
4. **▶ Hacer todo**. Los resultados quedan en la subcarpeta `salida`:
   - Videos: `... [EN].mp4` (subtítulos seleccionables), `... [EN] subs ES.mp4` (subtítulos grabados) y los `.srt`.
   - Audios: `... [EN].mp3` con la voz doblada, más los `.srt` del original y de la traducción.

Los subtítulos van en castellano cuando se puede (si el original es castellano, o si se dobla al castellano);
si no, en el idioma original. Si el audio ya está en el idioma elegido (por ejemplo castellano → castellano),
no se traduce: solo se cambia la voz.

En **Opciones avanzadas** podés fijar el idioma original (si la detección automática se equivoca) y contar
de qué trata el material (quién habla y sobre qué): ayuda a traducir mejor y queda guardado para esa carpeta.
Para una serie con nombres difíciles podés dejar un `glosario.json` en `<carpeta>/_proceso/` (mismo formato que
`app/glosario.json`): arreglos fijos de la transcripción, nombres propios y cómo pronunciar palabras.

Mientras trabaja, cada video muestra su barra de avance y abajo el tiempo que falta.
- **⏸ Pausar** frena todo y libera la placa de video y el procesador; **▶ Continuar** sigue en el mismo punto.
- **⏹ Detener** corta y libera también la memoria; después, **Hacer todo** sigue desde donde quedó.

No pongas la muestra de voz dentro de la carpeta de los videos: se doblaría como un audio más.

## 🌐 Traductores
En **Opciones avanzadas → Traductor** elegís quién traduce:
- **Qwen3-4B** (el de siempre): rápido, ya viene instalado.
- **Qwen3.5-9B**: más inteligente; corrige la transcripción, traduce y adapta las letras de las canciones.
- **Hy-MT2-7B** (Tencent): un modelo hecho solo para traducir; lo hablado lo traduce él, y la corrección y las
  letras las hace Qwen3-4B.

Los dos nuevos se instalan una sola vez desde Pinokio con **Instalar traductores** (corren en su propio entorno,
`env_traductor`, en 4 bits para que entren en la placa). La primera vez que usás cada uno baja su modelo
(Hy-MT2-7B ~16 GB, Qwen3.5-9B ~19 GB). Al cambiar de traductor se rehacen solas las traducciones automáticas;
las que revisaste a mano no se tocan. Para comparar, copiá el video doblado antes de cambiar de traductor.

## ✍️ Texto para tu voz
Abajo de todo está el cuadro **Texto para tu voz**: escribí, pegá o cargá un texto (.txt, .docx, .md o .srt) y
tocá **Generar audio con mi voz**. Lo dice tu voz clonada (la muestra de voz de la carpeta elegida, en
*3 · Voz → Muestra de voz propia*) y sale un mp3 en `salida/textos`, que también se escucha ahí mismo.
- Lo dice frase por frase y escucha cada una con Whisper: si una no se entiende, la repite.
- Una línea en blanco entre párrafos hace una pausa más larga.
- El idioma del texto se detecta solo (o elegilo): castellano, inglés, alemán, ruso, francés, italiano o portugués.
- Si cambiás algo y generás de nuevo, solo se rehacen las frases que cambiaron.

## 🎵 Canciones (opcional)
Si un video o audio tiene partes cantadas, la app las detecta sola y las **vuelve a cantar en el idioma del
doblaje, con la misma melodía y el timbre del cantante** (YingMusic-Singer-Plus). Lo hablado se dobla como siempre.

En las canciones manda la música: **la voz tiene que caer en las notas, no al revés**. Por eso, además de cantar,
la app usa un **conversor de voz cantada** (YingMusic-SVC) que toma las notas, el ritmo y el vibrato del cantante
original y solo cambia quién canta:

- **Canción ya en el idioma del doblaje + voz de una muestra:** no se vuelve a cantar; se convierte la voz original a
  la de la muestra. Suena como el cantante, pero con tu voz.
- **Canción traducida:** se canta la letra nueva con el timbre del cantante (nunca con la muestra hablada, que la hacía
  sonar recitada) y el conversor la hace caer en las notas exactas de la melodía original, con la voz de la muestra
  si la hay.

1. En Pinokio tocá **🎵 Instalar canto** (una sola vez, ~11 GB más con el conversor; el modelo de canto, ~3 GB, se
   baja la primera vez que canta). Si ya lo tenías instalado, aparece **🎵 Actualizar canto** para sumar el conversor.
2. En la interfaz dejá tildado **Cantar las canciones** y tocá **▶ Hacer todo**.

- La letra se traduce buscando la misma cantidad de sílabas por línea, para que entre en las notas.
- Canta 3 versiones de cada trozo y se queda con la que mejor se entiende (la escucha con Whisper) y más afinada.
- **Notas exactas:** el conversor lleva la voz nueva a la melodía del cantante original (con su vibrato), en la
  octava que le queda cómoda a la voz destino, y se le da sonido de voz cantada (presencia, compresión suave y un
  poco de sala). Sin el conversor se afina con WORLD como antes (se puede apagar en Opciones avanzadas).
- Las partes cantadas van a velocidad normal (el video no se acelera ahí) y la música no se baja.
- Si una canción ya está en el idioma del doblaje y la voz es la del video, queda la original.
- El inglés es el que mejor canta; alemán, ruso y castellano salen con pronunciación aproximada (experimental).
- Si algún trozo no se puede cantar, queda la voz original en ese trozo.
- En cada video aparece 🎵 con la cantidad de partes cantadas que encontró.

La primera vez se descargan los modelos (Whisper, Demucs, Qwen3-TTS y el traductor local, unos 16 GB).
En `_proceso/capNN/revisar_XX.txt` queda una lista de lo dudoso de cada video, para revisarlo.

Importante: cloná solo voces de personas que te hayan dado permiso.

## Créditos
Demucs (Meta), Whisper (OpenAI), Qwen3-TTS, Qwen3 y Qwen3.5 (Alibaba), Hy-MT2 (Tencent, Apache 2.0), YingMusic-Singer-Plus (ASLP-lab, CC BY 4.0; su VAE
es de Stable Audio Open, Stability AI Community License), YingMusic-SVC (GiantAILab, código MIT; usa RMVPE,
CAM++ de FunASR, BigVGAN v2 de NVIDIA y Whisper), detector de sonidos AST (MIT), espeak-ng, y las capas
de descript-audio-codec (MIT) en `app/canto/dac_min`. Letra de la interfaz: Lexend (SIL Open Font License,
`app/fuentes/OFL.txt`).
