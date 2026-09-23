"""
ECHO voice assistant pipeline.

Architecture:

    Microphone
        ↓
    openWakeWord
        ↓
    "Alexa"
        ↓
    "I am listening."
        ↓
    Browser confirms audio finished
        ↓
    VAD
        ↓
    Whisper
        ↓
    Gemma / llama.cpp
        ↓
    Tools
        ↓
    Kokoro
        ↓
    Browser
        ↓
    Browser confirms audio finished
        ↓
    LISTENING

Important:

    Alexa is detected by openWakeWord.

    Whisper is used only for user transcription.

    Python does NOT assume when browser audio finishes.
    The browser sends an "audio_finished" event.
"""


# ============================================================
# STANDARD LIBRARY
# ============================================================

import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
import types


# ============================================================
# NUMPY
# ============================================================

import numpy as np


# ============================================================
# AUDIO
# ============================================================

import sounddevice as sd
import webrtcvad


# ============================================================
# HTTP
# ============================================================

import requests


# ============================================================
# WAV
# ============================================================

from scipy.io.wavfile import write


# ============================================================
# OPENAI CLIENT
# ============================================================

from openai import OpenAI


# ============================================================
# LITERT COMPATIBILITY BRIDGE
# ============================================================
#
# Your installed openWakeWord version expects:
#
#     tflite_runtime.interpreter
#
# Your Apple Silicon environment provides:
#
#     ai_edge_litert.interpreter
#
# This bridge allows the installed openWakeWord version to
# use LiteRT without requiring the unavailable old
# tflite-runtime package.
# ============================================================

try:

    from ai_edge_litert import (
        interpreter as litert_interpreter
    )

except ImportError as exc:

    raise RuntimeError(
        "ai-edge-litert is not installed. "
        "Run: pip install ai-edge-litert"
    ) from exc


tflite_runtime_module = types.ModuleType(
    "tflite_runtime"
)


tflite_runtime_module.__path__ = []


sys.modules[
    "tflite_runtime"
] = tflite_runtime_module


sys.modules[
    "tflite_runtime.interpreter"
] = litert_interpreter


# ============================================================
# OPENWAKEWORD
# ============================================================

from openwakeword.model import Model


# ============================================================
# MLX
# ============================================================

import mlx_whisper

from mlx_audio.tts.utils import load_model


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000


VAD_FRAME_MS = 30


VAD_AGGRESSIVENESS = 2


START_TIMEOUT_SECONDS = 8.0


SILENCE_DURATION_SECONDS = 0.5


MAX_RECORD_SECONDS = 30.0


MIN_SPEECH_SECONDS = 0.25


# ============================================================
# WAKE WORD CONFIGURATION
# ============================================================

WAKEWORD = "alexa"


EXIT_WORD = "stop"


WAKE_FRAME_LENGTH = 1280


WAKE_THRESHOLD = 0.65


WAKE_COOLDOWN_SECONDS = 1.5


WAKE_TIMEOUT_SECONDS = 60.0


WAKE_LOG_INTERVAL_SECONDS = 0.5


# ============================================================
# ACOUSTIC PROTECTION
# ============================================================

WAKE_POST_ACK_DELAY_SECONDS = 0.35


POST_WAKE_SILENCE_SECONDS = 0.35


# ============================================================
# MODELS
# ============================================================

WHISPER_MODEL = (
    "mlx-community/whisper-small-mlx"
)


KOKORO_MODEL_NAME = (
    "mlx-community/Kokoro-82M-bf16"
)


KOKORO_VOICE = "af_heart"


KOKORO_LANG_CODE = "a"


# ============================================================
# LLAMA.CPP / GEMMA
# ============================================================

LLM_BASE_URL = (
    "http://localhost:8080/v1"
)


LLM_MODEL = (
    "gemma-4-e4b/"
    "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
)


SYSTEM_PROMPT = (
    "You are a helpful bilingual voice assistant. "
    "Respond in the same language the user speaks. "
    "If the user speaks English, respond in English. "
    "If the user speaks Spanish, respond in Spanish. "
    "Do not translate unless the user explicitly asks you to. "
    "Keep replies short and conversational, like you're speaking "
    "out loud - a sentence or two."
)


llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="not-needed",
)


# ============================================================
# TOOLS
# ============================================================

TOOLS = [

    {
        "type": "function",

        "function": {

            "name":
                "get_current_time",

            "description": (
                "Get the current date and time. "
                "Use this when the user asks what time "
                "or date it is."
            ),

            "parameters": {

                "type":
                    "object",

                "properties":
                    {},

                "required":
                    [],
            },
        },
    },


    {
        "type": "function",

        "function": {

            "name":
                "greet_choco",

            "description": (
                "Use this tool ONLY when the user explicitly "
                "asks you to greet Choco. Examples: "
                "'saluda a Choco', "
                "'dile hola a Choco', "
                "'say hello to Choco'."
            ),

            "parameters": {

                "type":
                    "object",

                "properties":
                    {},

                "required":
                    [],
            },
        },
    },


    {
        "type": "function",

        "function": {

            "name":
                "get_weather_garcia",

            "description": (
                "Get the current weather conditions in "
                "García, Nuevo León, Mexico. "
                "Use this when the user asks about "
                "weather, temperature, rain, humidity, "
                "wind, or current conditions there."
            ),

            "parameters": {

                "type":
                    "object",

                "properties":
                    {},

                "required":
                    [],
            },
        },
    },
]


# ============================================================
# TOOLS IMPLEMENTATION
# ============================================================

def get_current_time() -> str:

    return time.strftime(
        "%A, %B %d, %Y at %I:%M %p"
    )


def greet_choco() -> str:

    return (
        "Hola Choco, somos tus amigos"
    )


def weather_code_to_spanish(
    code,
) -> str:

    descriptions = {

        0:
            "cielo despejado",

        1:
            "principalmente despejado",

        2:
            "parcialmente nublado",

        3:
            "nublado",

        45:
            "niebla",

        48:
            "niebla con escarcha",

        51:
            "llovizna ligera",

        53:
            "llovizna moderada",

        55:
            "llovizna intensa",

        56:
            "llovizna helada ligera",

        57:
            "llovizna helada intensa",

        61:
            "lluvia ligera",

        63:
            "lluvia moderada",

        65:
            "lluvia intensa",

        66:
            "lluvia helada ligera",

        67:
            "lluvia helada intensa",

        71:
            "nevada ligera",

        73:
            "nevada moderada",

        75:
            "nevada intensa",

        77:
            "granos de nieve",

        80:
            "chubascos ligeros",

        81:
            "chubascos moderados",

        82:
            "chubascos intensos",

        85:
            "chubascos de nieve ligeros",

        86:
            "chubascos de nieve intensos",

        95:
            "tormenta eléctrica",

        96:
            "tormenta eléctrica con granizo ligero",

        99:
            "tormenta eléctrica con granizo intenso",
    }


    return descriptions.get(
        code,
        "condiciones meteorológicas desconocidas",
    )


def get_weather_garcia() -> str:

    latitude = 25.8069

    longitude = -100.6198


    url = (
        "https://api.open-meteo.com/v1/forecast"
    )


    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "weather_code,"
            "wind_speed_10m"
        ),

        "timezone":
            "America/Monterrey",
    }


    response = requests.get(
        url,
        params=params,
        timeout=10,
    )


    response.raise_for_status()


    data = response.json()


    current =data["current"]


    temperature = current.get(
        "temperature_2m"
    )


    humidity = current.get(
        "relative_humidity_2m"
    )


    apparent = current.get(
        "apparent_temperature"
    )


    wind = current.get(
        "wind_speed_10m"
    )


    weather_code = current.get(
        "weather_code"
    )


    description =weather_code_to_spanish(
            weather_code
        )


    return (
        f"En García, Nuevo León, actualmente "
        f"hay {temperature} grados Celsius, "
        f"{description}. "
        f"La sensación térmica es de "
        f"{apparent} grados, "
        f"la humedad es de {humidity} por ciento "
        f"y el viento es de {wind} kilómetros por hora."
    )


# ============================================================
# AVAILABLE TOOLS
# ============================================================

AVAILABLE_TOOLS = {

    "get_current_time":
        get_current_time,

    "greet_choco":
        greet_choco,

    "get_weather_garcia":
        get_weather_garcia,
}


# ============================================================
# CONVERSATION
# ============================================================

conversation = [

    {
        "role":
            "system",

        "content":
            SYSTEM_PROMPT,
    }

]


# ============================================================
# MODEL OBJECTS
# ============================================================

kokoro_model = None

wakeword_model = None

audio_sequence = 0


# ============================================================
# LOAD MODELS
# ============================================================

def load_models() -> None:

    global kokoro_model
    global wakeword_model


    print(
        "Loading Kokoro model..."
    )


    kokoro_model = load_model(
        KOKORO_MODEL_NAME
    )


    print(
        "Kokoro model loaded."
    )


    print(
        f'Loading openWakeWord "{WAKEWORD}"...'
    )


    wakeword_model = Model(
        wakeword_models=[
            WAKEWORD
        ],

        inference_framework=
            "tflite",
    )


    print(
        f'openWakeWord "{WAKEWORD}" loaded.'
    )


# ============================================================
# WAV
# ============================================================

def _pcm_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
) -> bytes:

    buffer = io.BytesIO()


    write(
        buffer,
        sample_rate,
        audio,
    )


    return buffer.getvalue()


# ============================================================
# WAKE WORD
# ============================================================

def detect_wake_phrase(
    emit,
    stop_event=None,
) -> bool:

    """
    Listen continuously for "Alexa".

    Whisper is NOT used here.
    """

    if wakeword_model is None:

        raise RuntimeError(
            "Wake-word model is not loaded."
        )


    wakeword_model.reset()


    start_time = time.time()


    last_detection_time = 0.0

    last_log_time = 0.0


    print(
        'Waiting for wake word "Alexa"...'
    )


    emit(
        "state",
        {
            "value":
                "waiting_wake"
        },
    )


    with sd.RawInputStream(

        samplerate=
            SAMPLE_RATE,

        blocksize=
            WAKE_FRAME_LENGTH,

        channels=1,

        dtype="int16",

        latency="high",

    ) as stream:


        while True:

            if (
                stop_event is not None
                and
                stop_event.is_set()
            ):

                return False


            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            if (
                time.time()
                -
                start_time
                >=
                WAKE_TIMEOUT_SECONDS
            ):

                print(
                    "Wake-word timeout."
                )

                return False


            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            data, overflowed = (
                stream.read(
                    WAKE_FRAME_LENGTH
                )
            )


            if overflowed:

                print(
                    "WARNING: microphone overflow "
                    "during wake detection."
                )


            # ------------------------------------------------
            # NumPy int16
            # ------------------------------------------------

            pcm = np.frombuffer(
                data,
                dtype=np.int16,
            ).copy()


            if (
                len(pcm)
                !=
                WAKE_FRAME_LENGTH
            ):

                continue


            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = (
                wakeword_model.predict(
                    pcm
                )
            )


            score = float(
                prediction.get(
                    WAKEWORD,
                    0.0
                )
            )


            microphone_rms = float(
                np.sqrt(
                    np.mean(
                        pcm.astype(
                            np.float32
                        ) ** 2
                    )
                )
            )


            now = time.time()


            # ------------------------------------------------
            # Debug score
            # ------------------------------------------------

            if (
                now -
                last_log_time
                >=
                WAKE_LOG_INTERVAL_SECONDS
            ):

                print(
                    f"\r"
                    f"{WAKEWORD}: "
                    f"{score:.4f} "
                    f"mic_rms: "
                    f"{microphone_rms:.1f}",
                    end="",
                    flush=True,
                )


                last_log_time = now


            # ------------------------------------------------
            # Detection + cooldown
            # ------------------------------------------------

            if (
                score >=
                WAKE_THRESHOLD
                and
                (
                    now -
                    last_detection_time
                )
                >=
                WAKE_COOLDOWN_SECONDS
            ):

                last_detection_time =now


                print()
                print()


                print(
                    "----------------------------------------"
                )


                print(
                    f'WAKE WORD DETECTED: '
                    f'"{WAKEWORD}"'
                )


                print(
                    f"Confidence: {score:.4f}"
                )


                print(
                    "----------------------------------------"
                )


                print()


                wakeword_model.reset()


                emit(
                    "state",
                    {
                        "value":
                            "wake"
                    },
                )


                return True


# ============================================================
# POST-WAKE SILENCE
# ============================================================

def wait_for_post_wake_silence() -> None:

    """
    Confirm that the microphone becomes quiet before
    opening the actual user recording stage.
    """

    vad = webrtcvad.Vad(
        VAD_AGGRESSIVENESS
    )


    frame_samples = int(
        SAMPLE_RATE
        *
        VAD_FRAME_MS
        /
        1000
    )


    required_silent_frames = max(
        1,

        int(
            POST_WAKE_SILENCE_SECONDS
            /
            (
                VAD_FRAME_MS /
                1000
            )
        ),
    )


    silent_frames = 0


    print(
        "Waiting for microphone silence..."
    )


    with sd.RawInputStream(

        samplerate=
            SAMPLE_RATE,

        blocksize=
            frame_samples,

        channels=1,

        dtype="int16",

        latency="high",

    ) as stream:


        while (
            silent_frames
            <
            required_silent_frames
        ):


            data, overflowed = (
                stream.read(
                    frame_samples
                )
            )


            if overflowed:

                print(
                    "WARNING: microphone overflow "
                    "during silence check."
                )


            frame =bytes(data)


            if len(frame) != (
                frame_samples * 2
            ):

                continue


            try:

                is_speech =vad.is_speech(
                        frame,
                        SAMPLE_RATE,
                    )

            except Exception:

                continue


            if is_speech:

                silent_frames =0

            else:

                silent_frames += 1


    print(
        "Microphone is quiet."
    )


# ============================================================
# USER VAD
# ============================================================

def _record_blocking_vad() -> np.ndarray:

    vad = webrtcvad.Vad(
        VAD_AGGRESSIVENESS
    )


    frame_samples = int(
        SAMPLE_RATE
        *
        VAD_FRAME_MS
        /
        1000
    )


    frame_bytes = (
        frame_samples * 2
    )


    silence_frames_required = max(
        1,

        int(
            SILENCE_DURATION_SECONDS
            /
            (
                VAD_FRAME_MS /
                1000
            )
        ),
    )


    min_speech_frames = max(
        1,

        int(
            MIN_SPEECH_SECONDS
            /
            (
                VAD_FRAME_MS /
                1000
            )
        ),
    )


    start_time = time.time()


    speech_started = False

    speech_frames = []

    silence_frames = 0

    speech_frame_count = 0


    print(
        "Listening for user speech..."
    )


    with sd.RawInputStream(

        samplerate=
            SAMPLE_RATE,

        blocksize=
            frame_samples,

        channels=1,

        dtype="int16",

        latency="high",

    ) as stream:


        while True:


            elapsed = (
                time.time()
                -
                start_time
            )


            if (
                elapsed
                >=
                MAX_RECORD_SECONDS
            ):

                print(
                    "Maximum recording time reached."
                )

                break


            data, overflowed = (
                stream.read(
                    frame_samples
                )
            )


            if overflowed:

                print(
                    "WARNING: microphone overflow."
                )


            frame =bytes(data)


            if len(frame) != frame_bytes:

                continue


            try:

                is_speech =vad.is_speech(
                        frame,
                        SAMPLE_RATE,
                    )

            except Exception as e:

                print(
                    f"VAD error: {e}"
                )

                continue


            # ------------------------------------------------
            # Speech detected
            # ------------------------------------------------

            if is_speech:

                if not speech_started:

                    speech_started = True

                    print(
                        "Speech detected."
                    )


                speech_frames.append(
                    np.frombuffer(
                        frame,
                        dtype=np.int16,
                    ).copy()
                )


                speech_frame_count += 1


                silence_frames =0


                continue


            # ------------------------------------------------
            # Silence after speech
            # ------------------------------------------------

            if speech_started:

                speech_frames.append(
                    np.frombuffer(
                        frame,
                        dtype=np.int16,
                    ).copy()
                )


                silence_frames += 1


                if (
                    silence_frames
                    >=
                    silence_frames_required
                ):

                    print(
                        "End of speech detected."
                    )

                    break


            else:

                if (
                    elapsed
                    >=
                    START_TIMEOUT_SECONDS
                ):

                    print(
                        "No speech detected."
                    )

                    break


    # ========================================================
    # Validate
    # ========================================================

    if (
        not speech_frames
        or
        speech_frame_count
        <
        min_speech_frames
    ):

        return np.array(
            [],
            dtype=np.int16,
        )


    # ========================================================
    # Combine
    # ========================================================

    audio =np.concatenate(
            speech_frames
        )


    # ========================================================
    # Remove trailing silence
    # ========================================================

    if silence_frames > 0:

        remove_samples = (
            silence_frames
            *
            frame_samples
        )


        if (
            remove_samples
            <
            len(audio)
        ):

            audio =audio[
                    :-remove_samples
                ]


    return audio


# ============================================================
# RECORD
# ============================================================

def record_audio(
    path: str = "turn_input.wav",
) -> float:

    start = time.time()


    audio =_record_blocking_vad()


    if audio.size == 0:

        audio =np.zeros(
                1,
                dtype=np.int16,
            )


    write(
        path,
        SAMPLE_RATE,
        audio,
    )


    elapsed =time.time() - start


    print(
        f"Recording finished: "
        f"{elapsed:.2f}s"
    )


    return elapsed


# ============================================================
# WHISPER
# ============================================================

def transcribe(
    path: str = "turn_input.wav",
) -> tuple[str, float]:

    start = time.time()


    result =mlx_whisper.transcribe(

            path,

            path_or_hf_repo=
                WHISPER_MODEL,

            task="transcribe",
        )


    text =result.get(
            "text",
            "",
        ).strip()


    elapsed =time.time() - start


    print(
        f"Whisper: {text}"
    )


    return (
        text,
        elapsed,
    )


# ============================================================
# GEMMA
# ============================================================

def get_reply(
    user_text: str,
    emit,
) -> tuple[str, float]:

    start = time.time()


    conversation.append(
        {
            "role":
                "user",

            "content":
                user_text,
        }
    )


    response =llm_client.chat.completions.create(

            model=
                LLM_MODEL,

            messages=
                conversation,

            tools=
                TOOLS,

            max_tokens=
                150,
        )


    message =response.choices[0].message


    # ========================================================
    # Normal response
    # ========================================================

    if not message.tool_calls:

        reply_text =message.content or ""


        if not reply_text:

            reply_text ="Sorry, I couldn't generate a response."


        conversation.append(
            {
                "role":
                    "assistant",

                "content":
                    reply_text,
            }
        )


        elapsed =time.time() - start


        print(
            f"Gemma: {reply_text}"
        )


        return (
            reply_text,
            elapsed,
        )


    # ========================================================
    # Tool calls
    # ========================================================

    conversation.append(
        message.model_dump()
    )


    for tool_call in message.tool_calls:

        fn_name =tool_call.function.name


        fn_args =json.loads(
                tool_call.function.arguments
                or "{}"
            )


        print(
            f"Tool requested: {fn_name}"
        )


        emit(
            "tool_call",
            {
                "name":
                    fn_name,

                "args":
                    fn_args,
            },
        )


        if fn_name not in AVAILABLE_TOOLS:

            raise RuntimeError(
                f"Unknown tool: {fn_name}"
            )


        result =AVAILABLE_TOOLS[
                fn_name
            ](
                **fn_args
            )


        print(
            f"Tool result: {result}"
        )


        # ====================================================
        # CHOCO — DIRECT
        # ====================================================

        if fn_name == "greet_choco":

            reply_text =str(result)


            conversation.append(
                {
                    "role":
                        "tool",

                    "tool_call_id":
                        tool_call.id,

                    "content":
                        reply_text,
                }
            )


            conversation.append(
                {
                    "role":
                        "assistant",

                    "content":
                        reply_text,
                }
            )


            elapsed =time.time() - start


            print(
                f"Choco direct response: "
                f"{reply_text}"
            )


            return (
                reply_text,
                elapsed,
            )


        # ====================================================
        # NORMAL TOOL
        # ====================================================

        conversation.append(
            {
                "role":
                    "tool",

                "tool_call_id":
                    tool_call.id,

                "content":
                    str(result),
            }
        )


    # ========================================================
    # Gemma follow-up
    # ========================================================

    follow_up =llm_client.chat.completions.create(

            model=
                LLM_MODEL,

            messages=
                conversation,

            max_tokens=
                150,
        )


    reply_text =follow_up.choices[0].message.content


    if not reply_text:

        reply_text ="Sorry, I couldn't generate a response."


    conversation.append(
        {
            "role":
                "assistant",

            "content":
                reply_text,
        }
    )


    elapsed =time.time() - start


    print(
        f"Gemma: {reply_text}"
    )


    return (
        reply_text,
        elapsed,
    )


# ============================================================
# KOKORO
# ============================================================

def synthesize(
    text: str,
) -> tuple[bytes, int, float, float]:

    start = time.time()


    if kokoro_model is None:

        raise RuntimeError(
            "Kokoro model has not been loaded."
        )


    audio_chunks = []

    sample_rate = 24000


    print(
        f"Kokoro synthesizing: {text}"
    )


    for result in (
        kokoro_model.generate(

            text=
                text,

            voice=
                KOKORO_VOICE,

            speed=
                1.0,

            lang_code=
                KOKORO_LANG_CODE,
        )
    ):

        audio_chunks.append(
            np.asarray(
                result.audio
            )
        )


        sample_rate =getattr(
                result,
                "sample_rate",
                sample_rate,
            )


    if not audio_chunks:

        raise RuntimeError(
            "Kokoro produced no audio."
        )


    audio =np.concatenate(
            audio_chunks
        )


    # ========================================================
    # Convert float → int16
    # ========================================================

    if np.issubdtype(
        audio.dtype,
        np.floating,
    ):

        audio =np.clip(
                audio,
                -1.0,
                1.0,
            )


        audio =(
                audio * 32767
            ).astype(
                np.int16
            )

    else:

        audio =audio.astype(
                np.int16
            )


    # ========================================================
    # WAV
    # ========================================================

    wav_data =_pcm_to_wav_bytes(
            audio,
            sample_rate,
        )


    # Debug copy

    with open(
        "turn_output.wav",
        "wb",
    ) as f:

        f.write(
            wav_data
        )


    duration =len(audio) / sample_rate


    elapsed =time.time() - start


    print(
        f"Kokoro generated "
        f"{len(wav_data)} bytes "
        f"at {sample_rate} Hz "
        f"({duration:.2f}s)"
    )


    return (
        wav_data,
        sample_rate,
        elapsed,
        duration,
    )


# ============================================================
# SEND AUDIO
# ============================================================

def send_audio(
    audio_data: bytes,
    sample_rate: int,
    emit,
) -> int:

    """
    Send WAV to browser.

    IMPORTANT:
        Caller must clear audio_finished_event BEFORE
        calling this function.
    """

    global audio_sequence


    audio_sequence += 1


    audio_id = audio_sequence


    audio_base64 =base64.b64encode(
            audio_data
        ).decode(
            "ascii"
        )


    emit(
        "audio",
        {
            "data":
                audio_base64,

            "mime":
                "audio/wav",

            "sample_rate":
                sample_rate,

            "audio_id":
                audio_id,
        },
    )


    return audio_id


# ============================================================
# WAIT FOR BROWSER PLAYBACK
# ============================================================

def wait_for_browser_audio(
    audio_finished_event,
    stop_event,
    playback_state,
    expected_audio_id,
    timeout=120.0,
) -> bool:

    """
    Wait until browser reports that audio playback has ended.

    IMPORTANT:

        The event must be cleared BEFORE audio is sent.

        We intentionally DO NOT clear it inside this function,
        because doing so could erase a very fast browser event.
    """

    print(
        "Waiting for browser to finish audio..."
    )


    start = time.time()


    while not stop_event.is_set():

        if audio_finished_event.wait(
            timeout=0.1
        ):

            if (
                playback_state.get(
                    "audio_id"
                )
                ==
                expected_audio_id
            ):

                print(
                    f"Browser audio finished "
                    f"(audio_id={expected_audio_id})."
                )

                return True


            audio_finished_event.clear()


        if (
            time.time() - start
            >= timeout
        ):

            print(
                "WARNING: timed out waiting "
                "for browser audio."
            )

            return False


    return False


# ============================================================
# WAKE ACK
# ============================================================

def speak_wake_acknowledgement(
    emit,
) -> tuple[bytes, int, float, float]:

    """
    Generate the wake acknowledgement audio.

    The caller is responsible for clearing the browser
    audio_finished event BEFORE calling this function.
    """

    text ="I am listening."


    emit(
        "wake_ack",
        {
            "text":
                text
        },
    )


    emit(
        "state",
        {
            "value":
                "speaking"
        },
    )


    (
        audio_data,
        sample_rate,
        t_tts,
        duration,
    ) = synthesize(
        text
    )


    return (
        audio_data,
        sample_rate,
        t_tts,
        duration,
    )


# ============================================================
# CONTINUOUS SESSION
# ============================================================

def run_session(
    emit,
    stop_event,
    audio_finished_event,
    playback_state,
    wait_for_wake_word: bool = True,
) -> None:

    """
    Continuous conversational session.

    ``wait_for_wake_word`` selects how the session starts. The normal default
    waits for ``Alexa``. The current browser connection uses this default.

    First:

        Alexa
        ↓
        I am listening.
        ↓
        user speech

    Then:

        response
        ↓
        browser playback
        ↓
        LISTENING
        ↓
        user speech again

    Alexa is required only once per session.
    """

    print()
    print(
                        "speaking"
    )
    print(
        "ECHO CONTINUOUS CONVERSATION STARTED"
    )
    print(
        "========================================"
    )
    print()


    try:

        # ====================================================
        # PHASE 1
        # WAKE WORD OR OPTIONAL DIRECT START
        # ====================================================

        if wait_for_wake_word:

            emit(
                "state",
                {
                    "value":
                        "waiting_wake"
                },
            )


            wake_detected =detect_wake_phrase(
                    emit,
                    stop_event,
                )


            if stop_event.is_set():

                return

            if not wake_detected:

                emit(
                    "state",
                    {
                        "value":
                            "idle"
                    },
                )

                return


        else:

            if stop_event.is_set():

                return


            print(
                "Direct session start activated. "
                "Skipping wake-word detection."
            )


        # ====================================================
        # PHASE 2
        # ALEXA DETECTED
        # ====================================================

        if wait_for_wake_word:

            print(
                'Alexa detected. Starting conversation.'
            )


        emit(
            "state",
            {
                "value":
                    "wake"
            },
        )


        # ====================================================
        # PHASE 3
        # WAKE ACK
        # ====================================================

        # IMPORTANT:
        # Clear BEFORE sending audio.

        audio_finished_event.clear()


        (
            wake_audio,
            wake_sample_rate,
            t_wake,
            wake_duration,
        ) = speak_wake_acknowledgement(
            emit
        )


        wake_audio_id = send_audio(
            wake_audio,
            wake_sample_rate,
            emit,
        )


        # ====================================================
        # WAIT FOR BROWSER
        # ====================================================

        wake_playback_finished = wait_for_browser_audio(
            audio_finished_event,
            stop_event,
            playback_state,
            wake_audio_id,
            timeout=max(
                wake_duration + 2.0,
                3.0,
            ),
        )


        if stop_event.is_set():

            return


        if not wake_playback_finished:

            print(
                "Browser did not confirm wake audio. "
                "Continuing after the duration fallback."
            )


        # ====================================================
        # SMALL ACOUSTIC DELAY
        # ====================================================

        time.sleep(
            WAKE_POST_ACK_DELAY_SECONDS
        )


        # ====================================================
        # CONFIRM MICROPHONE IS QUIET
        # ====================================================

        wait_for_post_wake_silence()


        if stop_event.is_set():

            return


        # ====================================================
        # FIRST LISTENING STATE
        # ====================================================

        emit(
            "state",
            {
                "value":
                    "listening"
            },
        )


        print(
            "ECHO is listening."
        )


        # ====================================================
        # CONTINUOUS CONVERSATION LOOP
        # ====================================================

        while not stop_event.is_set():

            try:

                # =================================================
                # LISTEN
                # =================================================

                emit(
                    "state",
                    {
                        "value":
                            "listening"
                    },
                )


                print(
                    "Listening for user speech..."
                )


                t_record =record_audio()


                if stop_event.is_set():

                    break


                # =================================================
                # TRANSCRIBING
                # =================================================

                emit(
                    "state",
                    {
                        "value":
                            "transcribing"
                    },
                )


                text, t_stt =transcribe()


                emit(
                    "transcript",
                    {
                        "text":
                            text
                    },
                )


                # =================================================
                # SPOKEN EXIT COMMAND
                # =================================================

                normalized_text = text.strip().lower().strip(
                    ".!?,'\""
                )


                if normalized_text == EXIT_WORD:

                    print(
                        'Exit word "stop" detected. '
                        "Ending conversation."
                    )

                    break


                # =================================================
                # Nothing heard
                # =================================================

                if not text:

                    print(
                        "Nothing heard. "
                        "Returning to listening."
                    )


                    emit(
                        "state",
                        {
                            "value":
                                "listening"
                        },
                    )


                    continue


                # =================================================
                # THINKING
                # =================================================

                emit(
                    "state",
                    {
                        "value":
                            "thinking"
                    },
                )


                reply, t_llm =get_reply(
                        text,
                        emit,
                    )


                emit(
                    "reply",
                    {
                        "text":
                            reply
                    },
                )


                if stop_event.is_set():

                    break


                # =================================================
                # KOKORO
                # =================================================

                emit(
                    "state",
                    {
                        "value":
                            "speaking"
                    },
                )


                (
                    audio_data,
                    sample_rate,
                    t_tts,
                    audio_duration,
                ) = synthesize(
                    reply
                )


                # =================================================
                # CRITICAL:
                #
                # Clear the completion event BEFORE sending the
                # audio.
                # =================================================

                audio_finished_event.clear()


                # =================================================
                # SEND AUDIO
                # =================================================

                reply_audio_id = send_audio(
                    audio_data,
                    sample_rate,
                    emit,
                )


                # =================================================
                # WAIT FOR REAL BROWSER PLAYBACK
                # =================================================

                playback_finished =wait_for_browser_audio(
                        audio_finished_event,
                        stop_event,
                    playback_state,
                    reply_audio_id,
                    )


                if stop_event.is_set():

                    break


                if not playback_finished:

                    print(
                        "Browser did not confirm "
                        "audio completion."
                    )


                # =================================================
                # SHORT ACOUSTIC SETTLING
                # =================================================

                time.sleep(
                    0.20
                )


                # =================================================
                # BACK TO LISTENING
                # =================================================

                if not stop_event.is_set():

                    emit(
                        "state",
                        {
                            "value":
                                "listening"
                        },
                    )


                    print(
                        "Response complete. "
                        "Listening for next question..."
                    )


                # =================================================
                # TIMING
                # =================================================

                emit(
                    "timing",
                    {
                        "record":
                            round(
                                t_record,
                                2,
                            ),

                        "stt":
                            round(
                                t_stt,
                                2,
                            ),

                        "llm":
                            round(
                                t_llm,
                                2,
                            ),

                        "tts":
                            round(
                                t_tts,
                                2,
                            ),

                        "play":
                            round(
                                audio_duration,
                                2,
                            ),
                    },
                )


            except Exception as turn_error:

                print(
                    f"Conversation turn error: "
                    f"{turn_error}"
                )


                emit(
                    "error",
                    {
                        "message":
                            str(turn_error)
                    },
                )


                if not stop_event.is_set():

                    emit(
                        "state",
                        {
                            "value":
                                "listening"
                        },
                    )


                time.sleep(
                    0.5
                )


    except Exception as e:

        print(
            f"Continuous session error: {e}"
        )


        emit(
            "error",
            {
                "message":
                    str(e)
            },
        )


    finally:

        emit(
            "state",
            {
                "value":
                    "idle"
            },
        )


        print()
        print(
            "ECHO continuous conversation stopped."
        )
        print()