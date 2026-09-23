# ECHO Voice Assistant

ECHO is a local FastAPI voice assistant for Apple Silicon. It listens for the
wake word `Alexa`, records a user utterance, transcribes it with Whisper, asks
a local Gemma/llama-server for a response, and speaks the response with
Kokoro.

The active application is a continuous voice assistant. It is no longer the
Phase 1 XTTS voice-cloning workflow described by the original README.

## Current Architecture

```text
Browser
    |
  | WebSocket: start_turn / audio_finished / stop
    v
FastAPI event loop
    |
    | run_in_executor
    v
Single mlx-worker thread
    |
    +--> Microphone and openWakeWord
    +--> WebRTC VAD and Whisper
    +--> Local Gemma through llama-server
    +--> Optional Python tools
    +--> Kokoro text-to-speech
    |
    v
Browser receives and plays a complete WAV response
```

The event-loop thread keeps the WebSocket responsive. The `mlx-worker` thread
runs blocking microphone and model operations. The executor intentionally has
`max_workers=1` because MLX GPU streams are thread-local in this application.
Do not increase the worker count without redesigning model ownership and
synchronization.

The pipeline is turn-buffered at the transcription, LLM, and TTS boundaries.
It sends complete Base64-encoded WAV events rather than streaming tokens or
audio chunks.

## Requirements

- Apple Silicon Mac with microphone and speaker access.
- Python environment with the dependencies required by `app.py` and
  `pipeline.py`.
- PortAudio for `sounddevice`.
- A local `llama-server` compatible with the OpenAI chat-completions API.
- The Gemma GGUF model referenced by `pipeline.py`.
- Permission for Terminal or the process running FastAPI to use the microphone.

The current `requirements.txt` is for the older TTS-focused environment and
does not completely describe every dependency used by the active application.
Install the missing runtime packages in your active environment as needed,
including FastAPI, Uvicorn, NumPy, SciPy, sounddevice, WebRTC VAD,
openWakeWord, LiteRT, mlx-whisper, mlx-audio, OpenAI, and requests.

## Setup

Install the system audio dependency:

```bash
brew install portaudio
```

Create or activate the Python environment used by this project. For example:

```bash
conda create -n voice-tts python=3.12 -y
conda activate voice-tts
conda install -c conda-forge ffmpeg -y
```

Install the active runtime dependencies in that environment. Package names can
vary by Apple Silicon and MLX version, so keep the environment consistent with
the versions already working on your machine.

## Local LLM Server

Start `llama-server` before asking ECHO questions. The application expects an
OpenAI-compatible endpoint at:

```text
http://localhost:8080/v1
```

The model identifier configured in `pipeline.py` is:

```text
gemma-4-e4b/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
```

The server must expose the model using the identifier expected by the
application, or `LLM_MODEL` in `pipeline.py` must be updated.

## Run ECHO

From the project directory, with the environment activated:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open this address in a browser:

```text
http://127.0.0.1:8000
```

During startup, `app.py` loads Kokoro and openWakeWord on the dedicated
`mlx-worker` thread. The browser then connects to `/ws`.

## User Flow

1. Open the ECHO page in the browser.
2. The browser connects to the WebSocket.
3. The server automatically starts one continuous session.
4. ECHO starts listening for the wake word.
5. Say `Alexa`.
6. ECHO plays `I am listening.`
7. The browser sends `audio_finished` after playback really ends.
8. ECHO waits for the microphone to become quiet, then records the question.
9. WebRTC VAD detects when the question begins and ends.
10. Whisper transcribes the recorded WAV.
11. Gemma generates a response or requests a tool.
12. Kokoro converts the response into WAV audio.
13. The browser plays the audio and sends another `audio_finished` event.
14. ECHO returns to listening for the next question without requiring `Alexa`
  again.
15. Say `stop` at any turn to end the conversation without sending that word
  to Gemma.

There is no required start button. Opening the page starts the wake-word
listener automatically.

For browser audio, click or tap the speaker icon after opening the page. This
unlocks Web Audio under the browser's autoplay policy; it does not start or
control the conversation.

The UI states follow the same sequence: `SAY "ALEXA"`, `WAKE DETECTED`,
`SPEAKING...` for the acknowledgement, then `LISTENING...` for the first user
question. After each answer it returns to `LISTENING...` until `stop` is heard.

There are two ways to end a session:

- Say `stop`. Whisper recognizes the spoken exit word and the pipeline returns
  to idle without generating a reply.
- Click the stop control in the browser. This sends the `stop` action, sets a
  thread-safe stop event, and the pipeline exits cooperatively between
  blocking operations.

## WebSocket Contract

### Browser actions

| Action | Purpose |
| --- | --- |
| `start_turn` | Legacy/manual action that starts a session if one is not already running. |
| `audio_finished` | Tell the worker that browser playback has ended. |
| `stop` | Request that the current session stop. |
| `ping` | Check that the WebSocket is reachable. |

### Server events

| Event | Purpose |
| --- | --- |
| `state` | Updates the UI state, such as `listening` or `thinking`. |
| `wake_ack` | Announces the wake acknowledgement text. |
| `transcript` | Contains the user's transcribed speech. |
| `reply` | Contains the assistant's text response. |
| `tool_call` | Reports a requested Python tool. |
| `audio` | Contains a complete Base64-encoded WAV response. |
| `timing` | Reports recording, STT, LLM, TTS, and playback timing. |
| `error` | Reports a pipeline or connection error. |

Audio events include an `audio_id`. The browser returns the same ID with
`audio_finished`, preventing delayed completion events from advancing the wrong
conversation turn.

## Models and Tools

- Wake word: openWakeWord model `alexa`.
- Speech-to-text: `mlx-community/whisper-small-mlx`.
- Text-to-speech: `mlx-community/Kokoro-82M-bf16`, voice `af_heart`.
- LLM: Gemma GGUF served through local `llama-server`.
- Weather: Open-Meteo API for Garcia, Nuevo Leon.
- Other tools: current date/time and a direct greeting for Choco.

Kokoro uses a preset voice. The current active pipeline does not clone a
user's voice with XTTS-v2.

## Project Files

| Path | Responsibility |
| --- | --- |
| `app.py` | FastAPI server, WebSocket, executor, and browser events. |
| `pipeline.py` | Audio, wake word, STT, LLM, tools, TTS, and session loop. |
| `static/index.html` | Browser UI markup. |
| `static/app.js` | WebSocket client and browser audio playback. |
| `static/style.css` | Browser UI styling. |
| `MODEL_STATE.md` | Snapshot of the current runtime architecture. |
| `pipeline_explaination.md` | Detailed walkthrough of `pipeline.py`. |
| `TESTS/` | Earlier component and performance experiments. |
| `gemma-4-e4b/` | Local Gemma GGUF model directory. |

## Troubleshooting

### No microphone or `PortAudioError`

Open macOS **System Settings > Privacy & Security > Microphone** and grant
access to Terminal, VS Code, or the process running Uvicorn. Confirm that the
correct input device is selected by the system.

### The application starts but LLM replies fail

Check that `llama-server` is running on port `8080` and that its model name
matches `LLM_MODEL` in `pipeline.py`.

### Models fail during startup

Confirm that the MLX, Kokoro, openWakeWord, and LiteRT packages are installed
in the same environment used to start Uvicorn. The openWakeWord compatibility
bridge requires `ai-edge-litert`.

### The assistant responds to its own audio

The browser must send `audio_finished` only from the audio playback completion
callback. The pipeline intentionally waits for this event before reopening the
microphone. The wake acknowledgement has a duration fallback if browser
autoplay prevents that event from arriving.

### No assistant audio is audible

Click or tap the speaker icon to satisfy the browser's Web Audio autoplay
policy. The wake-word listener remains automatic; this control only unlocks
playback.

### First startup is slow

Model files may be downloaded and cached on first use. Later starts should
reuse the local cache.

### Port 8000 remains in use after stopping

Use `Ctrl+C` to stop Uvicorn. Do not use `Ctrl+Z`: it suspends the process,
but the suspended process can continue holding port `8000`.

To find and stop a stale listener:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <PID>
```

If the process is suspended and does not respond to the normal signal:

```bash
kill -9 <PID>
```

You can also inspect suspended shell jobs with:

```bash
jobs -l
```

## Development Notes

- Conversation history is module-level state, so the current runtime is
  intended for one local user.
- Microphone capture, Whisper, LLM requests, and Kokoro synthesis are blocking
  operations and run on the single MLX worker.
- `emit()` schedules WebSocket sends back onto the asyncio event loop because
  the worker must not call the async WebSocket directly.
- `audio_finished_event` is cleared before each new audio event to prevent a
  completion signal from an earlier response being reused.

For a function-by-function explanation, read
[`pipeline_explaination.md`](pipeline_explaination.md).