"""FastAPI server for the ECHO voice assistant.

This module has two cooperating execution contexts:

* The asyncio event-loop thread owns FastAPI and the WebSocket. It receives
    browser messages and schedules WebSocket sends.
* The single ``mlx-worker`` thread owns model loading and the blocking voice
    pipeline. MLX GPU streams are thread-local, so all MLX work must stay on
    this one thread.

The worker is started with :func:`asyncio.loop.run_in_executor`. It can send
events back to the browser through ``emit``; ``emit`` uses
:func:`asyncio.run_coroutine_threadsafe` because WebSocket sends belong to the
asyncio loop, not to the worker thread. The worker can also pause until the
browser reports that audio playback has ended by waiting on
``audio_finished_event``.

There are no custom classes in this file. ``FastAPI`` and ``WebSocket`` are
classes supplied by external libraries; this module configures and uses them.
"""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline


# ============================================================
# PATHS
# ============================================================

STATIC_DIR = Path(__file__).parent / "static"


# ============================================================
# MLX EXECUTOR
# ============================================================
#
# MLX GPU streams are thread-local.
#
# Keep ALL MLX work on one worker:
#
#   - Kokoro loading
#   - openWakeWord model loading
#   - Whisper
#   - Kokoro inference
#
# Do not increase this above 1.
# ============================================================

ml_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mlx-worker",
)


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models before serving requests and stop the MLX worker on exit.

    ``pipeline.load_models`` is blocking, so it is submitted to the same
    single-worker executor used later by conversation sessions. This keeps
    model creation and inference on one MLX-compatible thread.
    """

    print()
    print("========================================")
    print("Starting ECHO...")
    print("========================================")

    loop = asyncio.get_running_loop()


    # --------------------------------------------------------
    # Load models on the dedicated MLX worker.
    # --------------------------------------------------------

    await loop.run_in_executor(
        ml_executor,
        pipeline.load_models,
    )


    print()
    print("Models loaded.")
    print("ECHO ready.")
    print()


    yield


    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    print()
    print("Shutting down ECHO...")


    ml_executor.shutdown(
        wait=True
    )


    print("ECHO stopped.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    lifespan=lifespan
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
async def index() -> FileResponse:
    """Return the browser application entry point."""

    return FileResponse(
        STATIC_DIR / "index.html"
    )


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def ws_endpoint(
    websocket: WebSocket,
) -> None:
    """Bridge one WebSocket connection to a single worker-thread session.

    The asyncio loop receives browser actions; the executor worker performs
    blocking audio and model work. ``emit`` schedules worker events back onto
    the loop, while ``audio_finished_event`` lets the worker wait for browser
    playback before continuing.
    """

    await websocket.accept()


    print(
        "WebSocket client connected."
    )


    loop = asyncio.get_running_loop()


    # ========================================================
    # SESSION STATE
    # ========================================================

    stop_event = None

    session_future = None


    # --------------------------------------------------------
    # This event is set by the browser when an audio response
    # has physically finished playing.
    # --------------------------------------------------------

    audio_finished_event = (
        threading.Event()
    )


    playback_state = {
        "audio_id":
            None,
    }

    # Set when the WebSocket is closing so the worker does not schedule
    # events against a connection that can no longer receive them.
    connection_closed = (
        threading.Event()
    )


    # ========================================================
    # EMIT
    # ========================================================

    def emit(
        event_type: str,
        payload: dict,
    ) -> None:

        """
        Send one server event to the browser from the worker thread.

        This function does not call ``websocket.send_text`` directly. The
        WebSocket belongs to the asyncio event loop, so a worker-thread call
        must use ``run_coroutine_threadsafe`` to submit the coroutine to that
        loop. The returned future is observed by a callback so send failures
        are logged instead of being silently discarded.
        """

        if connection_closed.is_set():

            return

        message = json.dumps(
            {
                "type": event_type,
                **payload,
            },
            ensure_ascii=False,
        )


        print(
            f"WS OUT -> {event_type} "
            f"({len(message)} bytes)"
        )


        try:

            future = (
                asyncio
                .run_coroutine_threadsafe(
                    websocket.send_text(
                        message
                    ),
                    loop,
                )
            )

        except Exception as e:

            print(
                f"Could not schedule "
                f"WebSocket send: {e}"
            )

            return


        def handle_result(
            completed_future,
        ):
            """Log the result of the WebSocket send on the event loop."""

            try:

                completed_future.result()

                print(
                    f"WS SENT -> "
                    f"{event_type}"
                )

            except Exception as e:

                if connection_closed.is_set():

                    return

                print(
                    f"WebSocket send error "
                    f"for {event_type}: "
                    f"{e}"
                )


        future.add_done_callback(
            handle_result
        )


    # ========================================================
    # START SESSION
    # ========================================================

    def start_session():
        """Start the blocking pipeline once, unless it is already running.

        ``run_in_executor`` returns immediately with an asyncio Future. The
        receive loop can therefore continue accepting browser actions while
        ``pipeline.run_session`` runs on ``mlx-worker``. The same event and
        callback objects are passed into the pipeline so it can stop and
        communicate with this WebSocket session.
        """

        nonlocal stop_event
        nonlocal session_future


        # ----------------------------------------------------
        # Don't start another session if one is running.
        # ----------------------------------------------------

        if (
            session_future
            and
            not session_future.done()
        ):

            print(
                "ECHO session already running."
            )

            return


        # ----------------------------------------------------
        # New stop event.
        # ----------------------------------------------------

        stop_event = (
            threading.Event()
        )


        # ----------------------------------------------------
        # Make sure there is no stale audio completion event.
        # ----------------------------------------------------

        audio_finished_event.clear()


        print(
            "Starting continuous conversation session..."
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # run_session() executes entirely on the dedicated
        # MLX worker.
        # ----------------------------------------------------

        session_future = (
            loop.run_in_executor(

                ml_executor,

                pipeline.run_session,

                emit,

                stop_event,

                audio_finished_event,

                playback_state,
            )
        )


        print(
            "Continuous conversation session launched."
        )


    # ========================================================
    # STOP SESSION
    # ========================================================

    def stop_session():
        """Request that the worker finish its current session.

        ``threading.Event`` is used because the pipeline reads it from the
        worker thread while this function is called by the event-loop thread.
        Setting the event is cooperative: the pipeline must check it at its
        blocking-operation boundaries and return.
        """

        if stop_event is not None:

            if not stop_event.is_set():

                print(
                    "Stopping ECHO session..."
                )

                stop_event.set()


    # Start listening for the wake word as soon as this browser connects.
    # The legacy start_turn action remains supported by the receive loop.
    start_session()


    # ========================================================
    # WEBSOCKET RECEIVE LOOP
    # ========================================================

    try:

        while True:

            raw_message = (
                await websocket.receive_text()
            )


            print(
                f"WS IN <- {raw_message}"
            )


            try:

                data = json.loads(
                    raw_message
                )

            except json.JSONDecodeError:

                await websocket.send_text(
                    json.dumps(
                        {
                            "type":
                                "error",

                            "message":
                                "Invalid JSON.",
                        },
                        ensure_ascii=False,
                    )
                )

                continue


            action = data.get(
                "action"
            )


            # =================================================
            # START SESSION
            # =================================================

            if action == "start_turn":

                start_session()


            # =================================================
            # BROWSER AUDIO FINISHED
            # =================================================
            #
            # This is extremely important.
            #
            # The browser sends this only after:
            #
            #     source.onended
            #
            # has fired.
            #
            # Therefore Python knows that the user's speakers
            # have really finished the response.
            # =================================================

            elif action == "audio_finished":

                print(
                    "Browser reports: "
                    "audio playback finished."
                )


                audio_id = data.get(
                    "audio_id"
                )


                if audio_id is not None:

                    playback_state[
                        "audio_id"
                    ] = audio_id

                    audio_finished_event.set()

                else:

                    print(
                        "Ignoring audio_finished "
                        "without audio_id."
                    )


            # =================================================
            # STOP
            # =================================================

            elif action == "stop":

                stop_session()


            # =================================================
            # PING
            # =================================================

            elif action == "ping":

                await websocket.send_text(
                    json.dumps(
                        {
                            "type":
                                "pong"
                        }
                    )
                )


            # =================================================
            # UNKNOWN
            # =================================================

            else:

                print(
                    f"Unknown WebSocket action: "
                    f"{action}"
                )


    except WebSocketDisconnect:

        connection_closed.set()

        print(
            "WebSocket client disconnected."
        )


        stop_session()


    except Exception as e:

        connection_closed.set()

        print(
            f"WebSocket endpoint error: {e}"
        )


        stop_session()


    finally:

        connection_closed.set()

        stop_session()


        print(
            "WebSocket cleanup complete."
        )