"use strict";

/* ============================================================
   ECHO — VOICE CORE
   Continuous conversation frontend
   ============================================================ */


/* ============================================================
   SVG
   ============================================================ */

const SVG_NS =
    "http://www.w3.org/2000/svg";

const CENTER =
    160;


/* ============================================================
   DOM
   ============================================================ */

const ring =
    document.getElementById("ring");

const ticksGroup =
    document.getElementById("ticks");

const barsGroup =
    document.getElementById("bars");

const stateLabel =
    document.getElementById("state-label");

const audioHint =
    document.getElementById("audio-hint");

let audioEnabled =
    localStorage.getItem(
        "echo-audio-enabled"
    ) === "true";

const transcriptLog =
    document.getElementById("transcript-log");

const replyLog =
    document.getElementById("reply-log");

const toolsEl =
    document.getElementById("tools");

const conversationBtn =
    document.getElementById("conversation-btn");

const conversationPanel =
    document.getElementById("conversation-panel");

const conversationClose =
    document.getElementById("conversation-close");


/* ============================================================
   SESSION
   ============================================================ */

let sessionActive =
    false;


let currentServerState =
    "idle";


/* ============================================================
   WEBSOCKET
   ============================================================ */

let ws =
    null;


let reconnectTimer =
    null;


/* ============================================================
   AUDIO
   ============================================================ */

let audioContext =
    null;


let browserAudioPlaying =
    false;


let audioSource =
    null;


/* ============================================================
   STATE LABELS
   ============================================================ */

const STATE_LABELS = {

    idle:
        'SAY "ALEXA"',

    waiting_wake:
        'SAY "ALEXA"',

    wake:
        "WAKE DETECTED",

    speaking:
        "SPEAKING...",

    listening:
        "LISTENING...",

    transcribing:
        "TRANSCRIBING...",

    thinking:
        "THINKING...",

    error:
        "ERROR",
};


/* ============================================================
   AUDIO CONTEXT
   ============================================================ */

function getAudioContext() {

    if (!audioContext) {

        const AudioContextClass =
            window.AudioContext ||
            window.webkitAudioContext;


        if (!AudioContextClass) {

            throw new Error(
                "Web Audio API is not supported."
            );
        }


        audioContext =
            new AudioContextClass();


        console.log(
            "AudioContext created:",
            audioContext.state
        );
    }


    return audioContext;
}


/* ============================================================
   UNLOCK AUDIO
   ============================================================ */

async function unlockAudio() {

    try {

        const context =
            getAudioContext();


        if (
            context.state !==
            "running"
        ) {

            await context.resume();
        }


        /*
         * Tiny silent buffer.
         *
         * This operation can originate from a browser user
         * gesture when the page is interacted with.
         */

        const buffer =
            context.createBuffer(
                1,
                1,
                context.sampleRate
            );


        const source =
            context.createBufferSource();


        source.buffer =
            buffer;


        source.connect(
            context.destination
        );


        source.start(0);


        console.log(
            "Audio unlocked:",
            context.state
        );


        audioEnabled =
            true;

        localStorage.setItem(
            "echo-audio-enabled",
            "true"
        );

        updateAudioControl();

    } catch (error) {

        console.error(
            "Audio unlock failed:",
            error
        );
    }
}


function updateAudioControl() {

    if (!audioHint) {

        return;
    }


    audioHint.textContent =
        audioEnabled
            ? "🔊"
            : "🔇";


    audioHint.setAttribute(
        "aria-pressed",
        String(audioEnabled)
    );


    audioHint.setAttribute(
        "aria-label",
        audioEnabled
            ? "Disable assistant audio"
            : "Enable assistant audio"
    );


    audioHint.setAttribute(
        "title",
        audioEnabled
            ? "Disable assistant audio"
            : "Enable assistant audio"
    );


    audioHint.classList.toggle(
        "active",
        audioEnabled
    );
}


if (audioHint) {

    audioHint.addEventListener(
        "pointerup",
        event => {

            event.preventDefault();

            if (
                audioEnabled
                &&
                audioContext
                &&
                audioContext.state ===
                    "running"
            ) {

                audioEnabled =
                    false;

                localStorage.setItem(
                    "echo-audio-enabled",
                    "false"
                );

                updateAudioControl();

                return;
            }


            unlockAudio();
        }
    );
}


updateAudioControl();


/* ============================================================
   POLAR
   ============================================================ */

function polar(
    cx,
    cy,
    r,
    deg
) {

    const rad =
        ((deg - 90) *
            Math.PI) /
        180;


    return [
        cx +
            r *
            Math.cos(rad),

        cy +
            r *
            Math.sin(rad)
    ];
}


/* ============================================================
   RADAR TICKS
   ============================================================ */

for (
    let i = 0;
    i < 60;
    i++
) {

    const deg =
        i * 6;


    const major =
        i % 5 === 0;


    const [x1, y1] =
        polar(
            CENTER,
            CENTER,
            150,
            deg
        );


    const [x2, y2] =
        polar(
            CENTER,
            CENTER,
            major
                ? 135
                : 141,
            deg
        );


    const line =
        document.createElementNS(
            SVG_NS,
            "line"
        );


    line.setAttribute(
        "x1",
        x1
    );


    line.setAttribute(
        "y1",
        y1
    );


    line.setAttribute(
        "x2",
        x2
    );


    line.setAttribute(
        "y2",
        y2
    );


    line.setAttribute(
        "class",
        major
            ? "tick tick-major"
            : "tick"
    );


    ticksGroup.appendChild(
        line
    );
}


/* ============================================================
   EQUALIZER BARS
   ============================================================ */

const BAR_COUNT =
    28;


for (
    let i = 0;
    i < BAR_COUNT;
    i++
) {

    const deg =
        i *
        (360 / BAR_COUNT);


    const [x, y] =
        polar(
            CENTER,
            CENTER,
            88,
            deg
        );


    const bar =
        document.createElementNS(
            SVG_NS,
            "rect"
        );


    bar.setAttribute(
        "x",
        -1.5
    );


    bar.setAttribute(
        "y",
        -4
    );


    bar.setAttribute(
        "width",
        3
    );


    bar.setAttribute(
        "height",
        8
    );


    bar.setAttribute(
        "class",
        "bar"
    );


    bar.style.transform =
        `translate(${x}px, ${y}px) rotate(${deg}deg)`;


    bar.style.animationDelay =
        `${(i % 7) * 0.08}s`;


    barsGroup.appendChild(
        bar
    );
}


/* ============================================================
   APPLY VISUAL STATE
   ============================================================ */

function applyVisualState(
    value
) {

    if (!value) {

        value =
            "idle";
    }


    /* --------------------------------------------------------
       Ring
       -------------------------------------------------------- */

    ring.setAttribute(
        "class",
        `ring state-${value}`
    );


    /* --------------------------------------------------------
       Label
       -------------------------------------------------------- */

    stateLabel.textContent =
        STATE_LABELS[value]
        ||
        value.toUpperCase();


    stateLabel.setAttribute(
        "class",
        `state-label state-${value}`
    );


}


/* ============================================================
   STATE HANDLER
   ============================================================ */

function setState(
    value
) {

    if (!value) {

        return;
    }


    currentServerState =
        value;


    console.log(
        "STATE:",
        value,
        "| session:",
        sessionActive,
        "| audio:",
        browserAudioPlaying
    );


    /*
     * Do not visually leave SPEAKING while audio is still
     * physically playing.
     */

    if (
        browserAudioPlaying
        &&
        value !== "speaking"
    ) {

        console.log(
            "Deferring state until audio ends:",
            value
        );


        return;
    }


    applyVisualState(
        value
    );
}


/* ============================================================
   CHAT LOG
   ============================================================ */

function addLogEntry(
    container,
    text
) {

    if (!container) {

        return;
    }


    container
        .querySelectorAll(
            ".log-empty"
        )
        .forEach(
            element =>
                element.remove()
        );


    const p =
        document.createElement(
            "p"
        );


    p.className =
        "log-entry";


    p.textContent =
        text || "";


    container.appendChild(
        p
    );


    container.scrollTop =
        container.scrollHeight;
}


/* ============================================================
   TOOL CHIP
   ============================================================ */

function addToolChip(
    name,
    args
) {

    if (!toolsEl) {

        return;
    }


    const chip =
        document.createElement(
            "div"
        );


    chip.className =
        "tool-chip";


    const argString =
        args &&
        Object.keys(args).length
            ? JSON.stringify(args)
            : "";


    chip.textContent =
        `⚙ ${name}(${argString})`;


    toolsEl.appendChild(
        chip
    );
}


/* ============================================================
   CONVERSATION
   ============================================================ */

function openConversation() {

    if (!conversationPanel) {

        return;
    }


    conversationPanel.classList.add(
        "open"
    );


    if (conversationBtn) {

        conversationBtn.setAttribute(
            "aria-expanded",
            "true"
        );
    }
}


function closeConversation() {

    if (!conversationPanel) {

        return;
    }


    conversationPanel.classList.remove(
        "open"
    );


    if (conversationBtn) {

        conversationBtn.setAttribute(
            "aria-expanded",
            "false"
        );
    }
}


if (conversationBtn) {

    conversationBtn.addEventListener(
        "click",
        () => {

            if (
                conversationPanel.classList.contains(
                    "open"
                )
            ) {

                closeConversation();

            } else {

                openConversation();
            }
        }
    );
}


if (conversationClose) {

    conversationClose.addEventListener(
        "click",
        closeConversation
    );
}


/* ============================================================
   PLAY AUDIO
   ============================================================ */

async function playAudio(
    base64Audio,
    audioId
) {

    try {

        if (!base64Audio) {

            console.error(
                "Empty audio data."
            );

            return;
        }


        console.log(
            "========== AUDIO =========="
        );


        console.log(
            "Audio event received."
        );


        /* ----------------------------------------------------
           AudioContext
           ---------------------------------------------------- */

        const context =
            getAudioContext();


        if (!audioEnabled) {

            console.log(
                "Assistant audio is disabled."
            );


            if (
                ws &&
                ws.readyState ===
                    WebSocket.OPEN
            ) {

                ws.send(
                    JSON.stringify(
                        {
                            action:
                                "audio_finished",

                            audio_id:
                                audioId,
                        }
                    )
                );
            }


            return;
        }


        if (
            context.state !==
            "running"
        ) {

            await context.resume();
        }


        /* ----------------------------------------------------
           Base64 -> bytes
           ---------------------------------------------------- */

        const binaryString =
            atob(base64Audio);


        const bytes =
            new Uint8Array(
                binaryString.length
            );


        for (
            let i = 0;
            i < binaryString.length;
            i++
        ) {

            bytes[i] =
                binaryString.charCodeAt(i);
        }


        console.log(
            "WAV bytes:",
            bytes.length
        );


        /* ----------------------------------------------------
           Decode
           ---------------------------------------------------- */

        const audioBuffer =
            await context.decodeAudioData(
                bytes.buffer.slice(0)
            );


        console.log(
            "Decoded audio:",
            audioBuffer.duration,
            "seconds"
        );


        /* ----------------------------------------------------
           Create source
           ---------------------------------------------------- */

        const source =
            context.createBufferSource();


        audioSource =
            source;


        source.buffer =
            audioBuffer;


        source.connect(
            context.destination
        );


        /* ====================================================
           AUDIO FINISHED
           ==================================================== */

        source.onended = () => {

            console.log(
                "Browser audio playback finished."
            );


            browserAudioPlaying =
                false;


            audioSource =
                null;


            /*
             * IMPORTANT:
             *
             * Tell FastAPI that the physical browser
             * playback really ended.
             */

            if (
                ws &&
                ws.readyState ===
                    WebSocket.OPEN
            ) {

                ws.send(
                    JSON.stringify(
                        {
                            action:
                                "audio_finished",

                            audio_id:
                                audioId,
                        }
                    )
                );


                console.log(
                    "Sent audio_finished to server."
                );
            }


            /*
             * For continuous conversation, immediately
             * return to LISTENING.
             */

            if (sessionActive) {

                applyVisualState(
                    "listening"
                );

            } else {

                applyVisualState(
                    "idle"
                );
            }
        };


        /* ----------------------------------------------------
           ACTUAL START
           ---------------------------------------------------- */

        browserAudioPlaying =
            true;


        applyVisualState(
            "speaking"
        );


        source.start(0);


        console.log(
            "Browser audio playback STARTED."
        );


    } catch (error) {

        console.error(
            "Audio playback error:",
            error
        );


        if (audioHint) {

            audioHint.textContent =
                "AUDIO BLOCKED - CLICK OR TAP ONCE TO ENABLE AUDIO";

            audioHint.classList.remove(
                "hidden"
            );
        }


        browserAudioPlaying =
            false;


        audioSource =
            null;


        /*
         * Tell Python that playback failed, otherwise the
         * backend could remain waiting forever.
         */

        if (
            ws &&
            ws.readyState ===
                WebSocket.OPEN
        ) {

            ws.send(
                JSON.stringify(
                    {
                        action:
                            "audio_finished",

                        audio_id:
                            audioId,
                    }
                )
            );
        }


        sessionActive =
            true;


        if (sessionActive) {

            applyVisualState(
                "waiting_wake"
            );

        } else {

            applyVisualState(
                "idle"
            );
        }
    }
}


/* ============================================================
   WEBSOCKET CONNECT
   ============================================================ */

function connect() {

    /*
     * Prevent duplicate connections.
     */

    if (
        ws &&
        (
            ws.readyState ===
                WebSocket.OPEN
            ||
            ws.readyState ===
                WebSocket.CONNECTING
        )
    ) {

        return;
    }


    console.log(
        "Connecting WebSocket..."
    );


    const protocol =
        location.protocol ===
            "https:"
            ? "wss:"
            : "ws:";


    const url =
        `${protocol}//${location.host}/ws`;


    console.log(
        "WebSocket URL:",
        url
    );


    try {

        ws =
            new WebSocket(
                url
            );

    } catch (error) {

        console.error(
            "WebSocket creation error:",
            error
        );


        scheduleReconnect();


        return;
    }


    /* ========================================================
       OPEN
       ======================================================== */

    ws.onopen = () => {

        console.log(
            "WebSocket connected."
        );


        sessionActive =
            true;


        applyVisualState(
            "waiting_wake"
        );


    };


    /* ========================================================
       MESSAGE
       ======================================================== */

    ws.onmessage = async (
        event
    ) => {

        let msg;


        try {

            msg =
                JSON.parse(
                    event.data
                );

        } catch (error) {

            console.error(
                "Invalid JSON from server:",
                event.data
            );

            return;
        }


        console.log(
            "Server event:",
            msg.type
        );


        switch (
            msg.type
        ) {


            /* ==============================================
               STATE
               ============================================== */

            case "state":

                setState(
                    msg.value
                );

                break;


            /* ==============================================
               WAKE ACK
               ============================================== */

            case "wake_ack":

                console.log(
                    "Wake acknowledgement:",
                    msg.text
                );

                /*
                 * Do not put this in the conversation log.
                 *
                 * The actual audio event will switch the ring
                 * to SPEAKING.
                 */

                break;


            /* ==============================================
               TRANSCRIPT
               ============================================== */

            case "transcript":

                addLogEntry(
                    transcriptLog,
                    msg.text ||
                        "(nothing heard)"
                );

                break;


            /* ==============================================
               REPLY
               ============================================== */

            case "reply":

                addLogEntry(
                    replyLog,
                    msg.text ||
                        ""
                );

                break;


            /* ==============================================
               AUDIO
               ============================================== */

            case "audio":

                await playAudio(
                    msg.data,
                    msg.audio_id
                );

                break;


            /* ==============================================
               TOOL CALL
               ============================================== */

            case "tool_call":

                addToolChip(
                    msg.name,
                    msg.args
                );

                break;


            /* ==============================================
               TIMING
               ============================================== */

            case "timing":

                console.log(
                    "Pipeline timing:",
                    msg
                );

                break;


            /* ==============================================
               ERROR
               ============================================== */

            case "error":

                addLogEntry(
                    transcriptLog,
                    `⚠ ${msg.message}`
                );

                console.error(
                    "Backend error:",
                    msg.message
                );

                break;


            /* ==============================================
               UNKNOWN
               ============================================== */

            default:

                console.warn(
                    "Unknown server event:",
                    msg
                );
        }
    };


    /* ========================================================
       ERROR
       ======================================================== */

    ws.onerror = (
        error
    ) => {

        console.error(
            "WebSocket error:",
            error
        );
    };


    /* ========================================================
       CLOSE
       ======================================================== */

    ws.onclose = (
        event
    ) => {

        console.warn(
            "WebSocket closed:",
            event.code,
            event.reason
        );


        /*
         * A lost WebSocket means the session is no longer
         * usable from the browser.
         */

        sessionActive =
            false;


        browserAudioPlaying =
            false;


        audioSource =
            null;


        applyVisualState(
            "idle"
        );


        scheduleReconnect();
    };
}


/* ============================================================
   RECONNECT
   ============================================================ */

function scheduleReconnect() {

    if (reconnectTimer) {

        return;
    }


    reconnectTimer =
        setTimeout(
            () => {

                reconnectTimer =
                    null;

                connect();

            },
            1500
        );
}


/* ============================================================
   INITIALIZE
   ============================================================ */

sessionActive =
    false;


browserAudioPlaying =
    false;


applyVisualState(
    "idle"
);


connect();