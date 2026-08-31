import asyncio
import os
import tempfile
import time

import uvicorn

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect
)

from fastapi.responses import (
    StreamingResponse,
    HTMLResponse
)

from pydantic import BaseModel

from rag import ask_chatbot
from stt import transcribe_audio_file
from tts import text_to_speech


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DesFlyer Voice Assistant API",
    description="RAG based Voice Assistant using Gemma 2B",
    version="4.0"
)


# ============================================================
# QUESTION MODEL
# ============================================================

class Question(BaseModel):

    question: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "DesFlyer Voice Assistant API is running",

        "endpoints": {

            "chat":
                "/chat",

            "chat_stream":
                "/chat/stream",

            "voice":
                "/voice",

            "websocket":
                "/ws/voice"
        }
    }


# ============================================================
# NORMAL CHAT
# ============================================================

@app.post("/chat")
def chat(data: Question):

    print("\n====================================")
    print("QUESTION:", data.question)
    print("====================================")

    try:

        start_time = time.time()

        answer = ask_chatbot(
            data.question
        )

        generation_time = (
            time.time() - start_time
        )

        print(
            f"⏱️ Generation time: "
            f"{generation_time:.2f} seconds"
        )

        print(
            "\n🤖 DesFlyer:",
            answer
        )

        return {

            "question":
                data.question,

            "answer":
                answer
        }

    except Exception as error:

        print(
            "\n❌ Chat error:",
            error
        )

        return {

            "question":
                data.question,

            "answer":
                "Sorry, an error occurred."
        }


# ============================================================
# STREAMING CHAT
# ============================================================

@app.post("/chat/stream")
def chat_stream(data: Question):

    try:

        answer = ask_chatbot(
            data.question
        )

        return StreamingResponse(

            iter([answer]),

            media_type="text/plain"
        )

    except Exception as error:

        print(
            "❌ Streaming error:",
            error
        )

        return StreamingResponse(

            iter([
                "Sorry, an error occurred."
            ]),

            media_type="text/plain"
        )


# ============================================================
# WEBSOCKET VOICE
# ============================================================

@app.websocket("/ws/voice")
async def websocket_voice(
    websocket: WebSocket
):

    await websocket.accept()

    print("\n====================================")
    print("🔌 WebSocket connected")
    print("====================================")

    # ========================================================
    # SERVER STATE
    # ========================================================

    processing = False

    assistant_speaking = False

    try:

        # ====================================================
        # INITIAL CONNECTION
        # ====================================================

        await websocket.send_text(
            "CONNECTED"
        )

        await websocket.send_text(
            "READY"
        )

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            message = await websocket.receive()

            # =================================================
            # TEXT MESSAGE
            # =================================================

            if "text" in message:

                command = (
                    message["text"]
                    .strip()
                    .lower()
                )

                print(
                    "\n📨 Command:",
                    command
                )

                # =============================================
                # START
                # =============================================

                if command in {
                    "start",
                    "voice",
                    "listen"
                }:

                    if processing:

                        print(
                            "⚠️ Server is currently processing."
                        )

                        await websocket.send_text(
                            "BUSY"
                        )

                        continue

                    processing = True

                    print(
                        "\n🎤 STATE: LISTENING"
                    )

                    await websocket.send_text(
                        "LISTENING"
                    )

                # =============================================
                # BARGE-IN
                # =============================================

                elif command == "barge_in":

                    print(
                        "\n🛑 BARGE-IN REQUEST RECEIVED"
                    )

                    # The browser has already stopped
                    # assistant audio.

                    assistant_speaking = False

                    # Important:
                    # processing remains True because the
                    # next binary message is the user's
                    # interrupted speech.

                    processing = True

                    await websocket.send_text(
                        "BARGE_IN_ACK"
                    )

                    print(
                        "🎤 Server ready for interrupted speech."
                    )

                # =============================================
                # STOP
                # =============================================

                elif command in {
                    "stop",
                    "exit",
                    "quit",
                    "bye"
                }:

                    print(
                        "\n🛑 Stop requested."
                    )

                    processing = False

                    assistant_speaking = False

                    await websocket.send_text(
                        "EXIT"
                    )

                    break

                # =============================================
                # PING
                # =============================================

                elif command == "ping":

                    await websocket.send_text(
                        "pong"
                    )

                # =============================================
                # EMPTY
                # =============================================

                elif command == "empty":

                    print(
                        "⚠️ Empty recording received."
                    )

                    processing = False

                    await websocket.send_text(
                        "NO_SPEECH"
                    )

                    await websocket.send_text(
                        "READY"
                    )

                # =============================================
                # UNKNOWN COMMAND
                # =============================================

                else:

                    print(
                        "⚠️ Unknown command:",
                        command
                    )

                    await websocket.send_text(
                        "UNKNOWN_COMMAND"
                    )

            # =================================================
            # BINARY AUDIO
            # =================================================

            elif "bytes" in message:

                audio_data = message["bytes"]

                if not audio_data:

                    print(
                        "⚠️ Empty audio received."
                    )

                    processing = False

                    await websocket.send_text(
                        "NO_SPEECH"
                    )

                    await websocket.send_text(
                        "READY"
                    )

                    continue

                print(
                    f"\n🎧 Browser audio received: "
                    f"{len(audio_data)} bytes"
                )

                # =================================================
                # PROCESSING
                # =================================================

                processing = True

                await websocket.send_text(
                    "PROCESSING"
                )

                print(
                    "\n⚙️ STATE: PROCESSING"
                )

                # =================================================
                # SAVE AUDIO
                # =================================================

                audio_file = None

                try:

                    with tempfile.NamedTemporaryFile(

                        delete=False,

                        suffix=".webm"

                    ) as temp_audio:

                        temp_audio.write(
                            audio_data
                        )

                        audio_file = (
                            temp_audio.name
                        )

                    print(
                        "💾 Browser audio saved:",
                        audio_file
                    )

                except Exception as error:

                    print(
                        "❌ Could not save audio:",
                        error
                    )

                    processing = False

                    await websocket.send_text(
                        "ERROR"
                    )

                    await websocket.send_text(
                        "READY"
                    )

                    continue

                # =================================================
                # STT
                # =================================================

                print(
                    "\n📝 Starting STT..."
                )

                stt_start = time.time()

                try:

                    user_text = (

                        await asyncio.to_thread(

                            transcribe_audio_file,

                            audio_file

                        )

                    )

                except Exception as error:

                    print(
                        "❌ STT error:",
                        error
                    )

                    user_text = ""

                stt_time = (
                    time.time() - stt_start
                )

                print(
                    f"⏱️ STT time: "
                    f"{stt_time:.2f} seconds"
                )

                # =================================================
                # DELETE TEMP AUDIO
                # =================================================

                try:

                    if (

                        audio_file

                        and

                        os.path.exists(
                            audio_file
                        )

                    ):

                        os.remove(
                            audio_file
                        )

                except Exception:

                    pass

                # =================================================
                # NO SPEECH
                # =================================================

                if not user_text:

                    print(
                        "⚠️ No valid speech detected."
                    )

                    processing = False

                    await websocket.send_text(
                        "NO_SPEECH"
                    )

                    await websocket.send_text(
                        "READY"
                    )

                    continue

                # =================================================
                # EXIT VOICE COMMAND
                # =================================================

                if user_text == "__EXIT__":

                    processing = False

                    await websocket.send_text(
                        "EXIT"
                    )

                    break

                # =================================================
                # USER TRANSCRIPTION
                # =================================================

                await websocket.send_text(
                    "USER:" + user_text
                )

                print(
                    "\n🎤 User:",
                    user_text
                )

                # =================================================
                # RAG
                # =================================================

                print(
                    "\n🔎 Starting RAG..."
                )

                rag_start = time.time()

                try:

                    answer = (

                        await asyncio.to_thread(

                            ask_chatbot,

                            user_text

                        )

                    )

                except Exception as error:

                    print(
                        "❌ RAG error:",
                        error
                    )

                    processing = False

                    await websocket.send_text(
                        "ERROR"
                    )

                    await websocket.send_text(
                        "READY"
                    )

                    continue

                rag_time = (
                    time.time() - rag_start
                )

                print(
                    f"⏱️ RAG time: "
                    f"{rag_time:.2f} seconds"
                )

                # =================================================
                # EMPTY ANSWER
                # =================================================

                if not answer:

                    processing = False

                    await websocket.send_text(
                        "ERROR"
                    )

                    await websocket.send_text(
                        "READY"
                    )

                    continue

                # =================================================
                # SEND ANSWER TEXT
                # =================================================

                await websocket.send_text(
                    "ANSWER:" + answer
                )

                print(
                    "\n🤖 DesFlyer:",
                    answer
                )

                # =================================================
                # TTS
                # =================================================

                await websocket.send_text(
                    "SPEAKING"
                )

                assistant_speaking = True

                print(
                    "\n🔊 STATE: SPEAKING"
                )

                tts_start = time.time()

                audio_output = None

                try:

                    audio_output = (

                        await asyncio.to_thread(

                            text_to_speech,

                            answer

                        )

                    )

                except Exception as error:

                    print(
                        "❌ TTS error:",
                        error
                    )

                tts_time = (
                    time.time() - tts_start
                )

                print(
                    f"⏱️ TTS time: "
                    f"{tts_time:.2f} seconds"
                )

                # =================================================
                # SEND TTS AUDIO
                # =================================================

                if (

                    audio_output

                    and

                    os.path.exists(
                        audio_output
                    )

                ):

                    try:

                        with open(

                            audio_output,

                            "rb"

                        ) as audio_file:

                            wav_data = (
                                audio_file.read()
                            )

                        print(
                            f"🔊 Sending WAV: "
                            f"{len(wav_data)} bytes"
                        )

                        await websocket.send_bytes(
                            wav_data
                        )

                        print(
                            "✅ Audio sent to browser."
                        )

                    except Exception as error:

                        print(
                            "❌ Could not send audio:",
                            error
                        )

                        assistant_speaking = False

                        await websocket.send_text(
                            "TTS_ERROR"
                        )

                    finally:

                        try:

                            if os.path.exists(
                                audio_output
                            ):

                                os.remove(
                                    audio_output
                                )

                        except Exception:

                            pass

                else:

                    print(
                        "⚠️ TTS did not create audio."
                    )

                    assistant_speaking = False

                    await websocket.send_text(
                        "TTS_ERROR"
                    )

                # =================================================
                # IMPORTANT
                # =================================================

                # Browser controls the actual playback.
                #
                # Browser sends the next command/audio when:
                #
                # 1. TTS finishes
                # 2. User interrupts TTS
                #
                processing = False

                print(
                    "\n⏳ Waiting for browser playback event..."
                )

            # =================================================
            # UNKNOWN WEBSOCKET MESSAGE
            # =================================================

            else:

                print(
                    "⚠️ Unknown WebSocket message."
                )

    except WebSocketDisconnect:

        print(
            "\n🔌 WebSocket client disconnected."
        )

    except Exception as error:

        print(
            "\n❌ WebSocket error:",
            error
        )

    finally:

        processing = False

        assistant_speaking = False

        print(
            "🔌 WebSocket connection closed."
        )


# ============================================================
# VOICE WEB PAGE
# ============================================================

@app.get(
    "/voice",
    response_class=HTMLResponse
)
def voice_page():

    return HTMLResponse(

        content=r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    DesFlyer Voice Assistant
</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #111827
        );

    color: white;

    min-height: 100vh;

    display: flex;

    align-items: center;

    justify-content: center;

    padding: 25px;
}

.container {

    width: 100%;

    max-width: 850px;

    background: #111827;

    border:
        1px solid
        #263244;

    border-radius: 20px;

    padding: 30px;

    box-shadow:
        0 20px 60px
        rgba(
            0,
            0,
            0,
            0.5
        );
}

.header {

    text-align: center;

    margin-bottom: 25px;
}

.header h1 {

    margin: 0;

    font-size: 30px;
}

.header p {

    color: #94a3b8;

    margin-top: 8px;
}

.status-card {

    text-align: center;

    background: #020617;

    border:
        1px solid
        #263244;

    border-radius: 16px;

    padding: 25px;

    margin-bottom: 25px;
}

#status-icon {

    font-size: 48px;

    margin-bottom: 10px;
}

#status {

    font-size: 25px;

    font-weight: bold;
}

#description {

    margin-top: 8px;

    color: #94a3b8;

    font-size: 14px;
}

.status-ready {
    color: #22c55e;
}

.status-listening {
    color: #38bdf8;
}

.status-processing {
    color: #f59e0b;
}

.status-speaking {
    color: #a78bfa;
}

.status-error {
    color: #ef4444;
}

.status-connected {
    color: #60a5fa;
}

.status-disconnected {
    color: #ef4444;
}

.buttons {

    display: flex;

    justify-content: center;

    gap: 12px;

    flex-wrap: wrap;

    margin-bottom: 25px;
}

button {

    border: none;

    border-radius: 10px;

    padding:
        13px
        22px;

    font-size: 16px;

    font-weight: bold;

    cursor: pointer;

    transition: 0.2s;
}

button:hover {

    transform:
        translateY(-2px);
}

button:disabled {

    opacity: 0.35;

    cursor: not-allowed;

    transform: none;
}

#connectButton {

    background: #2563eb;

    color: white;
}

#startButton {

    background: #16a34a;

    color: white;
}

#stopButton {

    background: #dc2626;

    color: white;
}

.pipeline {

    display: flex;

    justify-content: center;

    align-items: center;

    gap: 7px;

    flex-wrap: wrap;

    margin-bottom: 25px;
}

.step {

    background: #1e293b;

    color: #94a3b8;

    padding:
        9px
        12px;

    border-radius: 8px;

    font-size: 13px;
}

.conversation-title {

    font-size: 18px;

    font-weight: bold;

    margin-bottom: 10px;
}

#output {

    background: #020617;

    border:
        1px solid
        #263244;

    border-radius: 14px;

    padding: 18px;

    min-height: 180px;

    max-height: 400px;

    overflow-y: auto;
}

.message {

    padding: 12px;

    margin-bottom: 12px;

    border-radius: 10px;

    line-height: 1.5;
}

.user {

    background: #172554;

    border-left:
        4px solid
        #3b82f6;
}

.assistant {

    background: #2e1065;

    border-left:
        4px solid
        #8b5cf6;
}

.system {

    color: #64748b;

    text-align: center;

    font-size: 13px;

    margin: 8px;
}

</style>

</head>

<body>

<div class="container">

    <div class="header">

        <h1>
            🎙️ DesFlyer Voice Assistant
        </h1>

        <p>
            Browser Voice → STT → RAG → TTS
        </p>

    </div>

    <div class="status-card">

        <div id="status-icon">
            🔴
        </div>

        <div
            id="status"
            class="status-disconnected"
        >
            Disconnected
        </div>

        <div id="description">

            Click Connect to start.

        </div>

    </div>

    <div class="buttons">

        <button
            id="connectButton"
            onclick="connectWebSocket()"
        >
            🔌 Connect
        </button>

        <button
            id="startButton"
            onclick="startVoice()"
            disabled
        >
            🎤 Start Voice
        </button>

        <button
            id="stopButton"
            onclick="stopVoice()"
            disabled
        >
            🛑 Stop
        </button>

    </div>

    <div class="pipeline">

        <div class="step">
            🎤 Browser Mic
        </div>

        <div>→</div>

        <div class="step">
            📡 WebSocket
        </div>

        <div>→</div>

        <div class="step">
            📝 Faster-Whisper
        </div>

        <div>→</div>

        <div class="step">
            🔎 RAG
        </div>

        <div>→</div>

        <div class="step">
            🤖 Gemma 2B
        </div>

        <div>→</div>

        <div class="step">
            🔊 TTS
        </div>

    </div>

    <div class="conversation-title">

        Conversation

    </div>

    <div id="output">

        <div class="system">

            Connect to start.

        </div>

    </div>

</div>

<script>


// ============================================================
// WEBSOCKET
// ============================================================

let socket = null;


// ============================================================
// MICROPHONE
// ============================================================

let audioStream = null;


// ============================================================
// MEDIA RECORDER
// ============================================================

let mediaRecorder = null;

let audioChunks = [];


// ============================================================
// STATE
// ============================================================

let isProcessing = false;

let isSpeaking = false;

let isRecording = false;

let stoppedByUser = false;


// ============================================================
// CURRENT AUDIO
// ============================================================

let currentAudio = null;

let currentAudioUrl = null;


// ============================================================
// NORMAL AUDIO ANALYSIS
// ============================================================

let audioContext = null;

let analyser = null;

let microphoneSource = null;

let silenceTimer = null;

let volumeCheckTimer = null;

let speechDetected = false;

let recordingStartTime = null;


// ============================================================
// BARGE-IN ANALYSIS
// ============================================================

let bargeInAudioContext = null;

let bargeInAnalyser = null;

let bargeInMicrophoneSource = null;

let bargeInCheckTimer = null;

let bargeInTriggered = false;

let bargeInSpeechStart = null;

let bargeInNoiseFloor = 0;


// ============================================================
// NORMAL RECORDING SETTINGS
// ============================================================

const SILENCE_THRESHOLD = 0.015;

const SILENCE_DURATION = 1500;

const MAX_RECORDING_TIME = 30000;


// ============================================================
// BARGE-IN SETTINGS
// ============================================================
//
// The old code used:
//
// BARGE_IN_THRESHOLD = 0.025
//
// This is too sensitive on some microphones.
//
// We now use an adaptive threshold.
//
// ============================================================

const BARGE_IN_MIN_THRESHOLD = 0.035;

const BARGE_IN_CONFIRMATION_TIME = 600;

const BARGE_IN_COOLDOWN = 350;

const BARGE_IN_SAMPLE_TIME = 500;


// ============================================================
// ELEMENTS
// ============================================================

const status =
    document.getElementById(
        "status"
    );

const icon =
    document.getElementById(
        "status-icon"
    );

const description =
    document.getElementById(
        "description"
    );

const connectButton =
    document.getElementById(
        "connectButton"
    );

const startButton =
    document.getElementById(
        "startButton"
    );

const stopButton =
    document.getElementById(
        "stopButton"
    );

const output =
    document.getElementById(
        "output"
    );


// ============================================================
// STATUS
// ============================================================

function setStatus(
    state,
    message
) {

    status.innerText =
        state;

    description.innerText =
        message;

    status.className = "";

    status.classList.add(
        "status-" +
        state.toLowerCase()
    );

    if (
        state === "Disconnected"
    ) {

        icon.innerText = "🔴";

    }

    else if (
        state === "Connected"
    ) {

        icon.innerText = "🔵";

    }

    else if (
        state === "Ready"
    ) {

        icon.innerText = "🟢";

    }

    else if (
        state === "Listening"
    ) {

        icon.innerText = "🎤";

    }

    else if (
        state === "Processing"
    ) {

        icon.innerText = "⚙️";

    }

    else if (
        state === "Speaking"
    ) {

        icon.innerText = "🔊";

    }

    else if (
        state === "Error"
    ) {

        icon.innerText = "❌";

    }

}


// ============================================================
// ADD MESSAGE
// ============================================================

function addMessage(
    text,
    type
) {

    const div =
        document.createElement(
            "div"
        );

    div.className =
        "message " +
        type;

    div.innerText =
        text;

    output.appendChild(
        div
    );

    output.scrollTop =
        output.scrollHeight;

}


// ============================================================
// STOP ASSISTANT AUDIO
// ============================================================

function stopAssistantAudio() {

    console.log(
        "🛑 Stopping assistant audio."
    );

    if (currentAudio) {

        try {

            currentAudio.pause();

            currentAudio.currentTime = 0;

        }

        catch (error) {

            console.warn(
                "Audio stop warning:",
                error
            );

        }

    }

    if (currentAudioUrl) {

        try {

            URL.revokeObjectURL(
                currentAudioUrl
            );

        }

        catch (error) {

            console.warn(
                error
            );

        }

    }

    currentAudio = null;

    currentAudioUrl = null;

    isSpeaking = false;

    stopBargeInDetection();

}


// ============================================================
// START BARGE-IN DETECTION
// ============================================================

async function startBargeInDetection() {

    if (!audioStream) {

        console.warn(
            "⚠️ No microphone stream."
        );

        return;

    }

    if (!isSpeaking) {

        return;

    }

    try {

        stopBargeInDetection();

        bargeInTriggered =
            false;

        bargeInSpeechStart =
            null;

        bargeInNoiseFloor =
            0;


        console.log(
            "🎤 Starting improved barge-in detection..."
        );


        bargeInAudioContext =
            new (
                window.AudioContext ||
                window.webkitAudioContext
            )();


        if (
            bargeInAudioContext.state ===
            "suspended"
        ) {

            await bargeInAudioContext.resume();

        }


        bargeInAnalyser =
            bargeInAudioContext.createAnalyser();


        bargeInAnalyser.fftSize =
            2048;


        bargeInAnalyser.smoothingTimeConstant =
            0.75;


        bargeInMicrophoneSource =
            bargeInAudioContext.createMediaStreamSource(
                audioStream
            );


        bargeInMicrophoneSource.connect(
            bargeInAnalyser
        );


        const dataArray =
            new Uint8Array(
                bargeInAnalyser.fftSize
            );


        // ====================================================
        // MEASURE BACKGROUND NOISE
        // ====================================================

        let noiseSamples = [];

        let noiseStart =
            Date.now();


        function calculateRMS() {

            bargeInAnalyser.getByteTimeDomainData(
                dataArray
            );

            let sum = 0;

            for (
                let i = 0;
                i < dataArray.length;
                i++
            ) {

                const normalized =
                    (
                        dataArray[i] -
                        128
                    ) / 128;

                sum +=
                    normalized *
                    normalized;

            }

            return Math.sqrt(
                sum /
                dataArray.length
            );

        }


        function checkVolume() {

            if (
                !isSpeaking
            ) {

                return;

            }

            if (
                !bargeInAnalyser
            ) {

                return;

            }


            const rms =
                calculateRMS();


            // =================================================
            // FIRST 500ms = NOISE CALIBRATION
            // =================================================

            if (
                Date.now() -
                noiseStart <
                BARGE_IN_SAMPLE_TIME
            ) {

                noiseSamples.push(
                    rms
                );

                bargeInCheckTimer =
                    requestAnimationFrame(
                        checkVolume
                    );

                return;

            }


            // =================================================
            // CALCULATE NOISE FLOOR
            // =================================================

            if (
                bargeInNoiseFloor === 0
            ) {

                if (
                    noiseSamples.length > 0
                ) {

                    const total =
                        noiseSamples.reduce(
                            (
                                a,
                                b
                            ) =>
                                a + b,
                            0
                        );

                    bargeInNoiseFloor =
                        total /
                        noiseSamples.length;

                }

                else {

                    bargeInNoiseFloor =
                        0.01;

                }


                console.log(
                    "🎚️ Barge-in noise floor:",
                    bargeInNoiseFloor.toFixed(4)
                );

            }


            // =================================================
            // ADAPTIVE THRESHOLD
            // =================================================

            const adaptiveThreshold =
                Math.max(

                    BARGE_IN_MIN_THRESHOLD,

                    bargeInNoiseFloor * 3

                );


            // =================================================
            // USER SPEECH
            // =================================================

            if (
                rms >
                adaptiveThreshold
            ) {

                if (
                    bargeInSpeechStart ===
                    null
                ) {

                    bargeInSpeechStart =
                        Date.now();

                    console.log(
                        "🎤 Possible user speech detected..."
                    );

                }


                const speechDuration =
                    Date.now() -
                    bargeInSpeechStart;


                // =================================================
                // REQUIRE CONTINUOUS SPEECH
                // =================================================

                if (

                    speechDuration >=
                    BARGE_IN_CONFIRMATION_TIME

                    &&

                    !bargeInTriggered

                ) {

                    console.log(
                        "🛑 Confirmed user speech."
                    );

                    triggerBargeIn();

                    return;

                }

            }

            else {

                // Noise / short sound.

                bargeInSpeechStart =
                    null;

            }


            bargeInCheckTimer =
                requestAnimationFrame(
                    checkVolume
                );

        }


        checkVolume();

    }

    catch (error) {

        console.error(
            "❌ Barge-in detection error:",
            error
        );

    }

}


// ============================================================
// TRIGGER BARGE-IN
// ============================================================

async function triggerBargeIn() {

    if (
        bargeInTriggered
    ) {

        return;

    }

    if (
        !isSpeaking
    ) {

        return;

    }


    bargeInTriggered =
        true;


    console.log(
        "\n🛑 USER BARGE-IN DETECTED"
    );


    // ========================================================
    // STOP BARGE-IN ANALYSIS FIRST
    // ========================================================

    stopBargeInDetection();


    // ========================================================
    // STOP TTS IMMEDIATELY
    // ========================================================

    stopAssistantAudio();


    // ========================================================
    // INFORM BACKEND
    // ========================================================

    if (

        socket &&

        socket.readyState ===
        WebSocket.OPEN

    ) {

        socket.send(
            "barge_in"
        );

    }


    addMessage(
        "🛑 Assistant interrupted.",
        "system"
    );


    setStatus(
        "Listening",
        "🎤 Assistant interrupted. Speak now..."
    );


    startButton.disabled =
        true;

    stopButton.disabled =
        false;


    isProcessing =
        true;


    // ========================================================
    // IMPORTANT DELAY
    // ========================================================
    //
    // Give the browser time to completely stop the TTS
    // playback before MediaRecorder starts.
    //
    // This prevents the first part of the recording from
    // containing audio playback noise.
    //
    // ========================================================

    await new Promise(
        resolve =>
            setTimeout(
                resolve,
                BARGE_IN_COOLDOWN
            )
    );


    if (
        stoppedByUser
    ) {

        return;

    }


    // ========================================================
    // START NEW RECORDING
    // ========================================================

    await startRecording();

}


// ============================================================
// STOP BARGE-IN DETECTION
// ============================================================

function stopBargeInDetection() {

    if (
        bargeInCheckTimer
    ) {

        cancelAnimationFrame(
            bargeInCheckTimer
        );

        bargeInCheckTimer =
            null;

    }


    if (
        bargeInMicrophoneSource
    ) {

        try {

            bargeInMicrophoneSource.disconnect();

        }

        catch (error) {

            console.warn(
                error
            );

        }

        bargeInMicrophoneSource =
            null;

    }


    if (
        bargeInAudioContext
    ) {

        try {

            bargeInAudioContext.close();

        }

        catch (error) {

            console.warn(
                error
            );

        }

        bargeInAudioContext =
            null;

    }


    bargeInAnalyser =
        null;

    bargeInSpeechStart =
        null;

    bargeInNoiseFloor =
        0;

}


// ============================================================
// CONNECT WEBSOCKET
// ============================================================

function connectWebSocket() {

    if (

        socket &&

        socket.readyState ===
        WebSocket.OPEN

    ) {

        setStatus(
            "Ready",
            "WebSocket is already connected."
        );

        return;

    }


    stoppedByUser =
        false;


    const protocol =
        window.location.protocol ===
        "https:"
        ? "wss:"
        : "ws:";


    const wsUrl =
        protocol +
        "//" +
        window.location.host +
        "/ws/voice";


    console.log(
        "🔌 WebSocket URL:",
        wsUrl
    );


    socket =
        new WebSocket(
            wsUrl
        );


    socket.binaryType =
        "arraybuffer";


    // ========================================================
    // OPEN
    // ========================================================

    socket.onopen =
        function() {

        console.log(
            "✅ WebSocket connected"
        );


        connectButton.disabled =
            true;

        startButton.disabled =
            false;

        stopButton.disabled =
            true;


        addMessage(
            "🔌 WebSocket connected",
            "system"
        );


        setStatus(
            "Connected",
            "Connection established."
        );

    };


    // ========================================================
    // MESSAGE
    // ========================================================

    socket.onmessage =
        async function(event) {


        // ====================================================
        // BINARY AUDIO
        // ====================================================

        if (
            event.data instanceof
            ArrayBuffer
        ) {

            console.log(
                "🔊 Audio received from server."
            );


            try {

                const audioBlob =
                    new Blob(
                        [
                            event.data
                        ],
                        {
                            type:
                                "audio/wav"
                        }
                    );


                currentAudioUrl =
                    URL.createObjectURL(
                        audioBlob
                    );


                const audio =
                    new Audio(
                        currentAudioUrl
                    );


                currentAudio =
                    audio;


                isSpeaking =
                    true;


                // =================================================
                // AUDIO ENDED
                // =================================================

                audio.onended =
                    async function() {

                    console.log(
                        "✅ TTS playback completed."
                    );


                    isSpeaking =
                        false;


                    currentAudio =
                        null;


                    if (
                        currentAudioUrl
                    ) {

                        URL.revokeObjectURL(
                            currentAudioUrl
                        );

                    }


                    currentAudioUrl =
                        null;


                    stopBargeInDetection();


                    if (
                        stoppedByUser
                    ) {

                        return;

                    }


                    // =============================================
                    // CONTINUOUS CONVERSATION
                    // =============================================

                    addMessage(
                        "🎤 Listening for your next question...",
                        "system"
                    );


                    setStatus(
                        "Listening",
                        "🎤 Speak naturally..."
                    );


                    startButton.disabled =
                        true;

                    stopButton.disabled =
                        false;


                    isProcessing =
                        true;


                    // =============================================
                    // SMALL TTS TAIL DELAY
                    // =============================================

                    await new Promise(
                        resolve =>
                            setTimeout(
                                resolve,
                                400
                            )
                    );


                    if (
                        stoppedByUser
                    ) {

                        return;

                    }


                    await startRecording();

                };


                // =================================================
                // AUDIO ERROR
                // =================================================

                audio.onerror =
                    function(error) {

                    console.error(
                        "❌ Audio playback error:",
                        error
                    );


                    isSpeaking =
                        false;


                    currentAudio =
                        null;


                    stopBargeInDetection();


                    setStatus(
                        "Error",
                        "⚠️ Could not play assistant audio."
                    );

                };


                // =================================================
                // PLAY AUDIO
                // =================================================

                await audio.play();


                console.log(
                    "▶️ Assistant audio playing."
                );


                setStatus(
                    "Speaking",
                    "🔊 Speaking... You can interrupt me anytime."
                );


                // =================================================
                // START BARGE-IN
                // =================================================

                await startBargeInDetection();

            }

            catch (error) {

                console.error(
                    "❌ Audio playback error:",
                    error
                );


                isSpeaking =
                    false;


                currentAudio =
                    null;


                stopBargeInDetection();


                setStatus(
                    "Error",
                    "⚠️ Could not play assistant audio."
                );

            }


            return;

        }


        // ====================================================
        // TEXT MESSAGE
        // ====================================================

        const message =
            event.data;


        console.log(
            "📨 Server:",
            message
        );


        // ====================================================
        // CONNECTED
        // ====================================================

        if (
            message === "CONNECTED"
        ) {

            setStatus(
                "Connected",
                "WebSocket connection established."
            );

        }


        // ====================================================
        // READY
        // ====================================================

        else if (
            message === "READY"
        ) {

            isProcessing =
                false;


            if (
                !isSpeaking
            ) {

                startButton.disabled =
                    false;

                stopButton.disabled =
                    true;


                setStatus(
                    "Ready",
                    "Click Start Voice to ask a question."
                );

            }

        }


        // ====================================================
        // LISTENING
        // ====================================================

        else if (
            message === "LISTENING"
        ) {

            isProcessing =
                true;


            startButton.disabled =
                true;

            stopButton.disabled =
                false;


            setStatus(
                "Listening",
                "🎤 Speak now... silence will stop recording."
            );


            await startRecording();

        }


        // ====================================================
        // PROCESSING
        // ====================================================

        else if (
            message === "PROCESSING"
        ) {

            isProcessing =
                true;


            startButton.disabled =
                true;

            stopButton.disabled =
                true;


            setStatus(
                "Processing",
                "⚙️ STT → RAG → TTS processing..."
            );

        }


        // ====================================================
        // USER
        // ====================================================

        else if (
            message.startsWith(
                "USER:"
            )
        ) {

            const text =
                message.substring(
                    5
                );


            addMessage(
                "🎤 You: " +
                text,
                "user"
            );

        }


        // ====================================================
        // ANSWER
        // ====================================================

        else if (
            message.startsWith(
                "ANSWER:"
            )
        ) {

            const answer =
                message.substring(
                    7
                );


            addMessage(
                "🤖 DesFlyer: " +
                answer,
                "assistant"
            );

        }


        // ====================================================
        // SPEAKING
        // ====================================================

        else if (
            message ===
            "SPEAKING"
        ) {

            isProcessing =
                true;

            isSpeaking =
                true;


            startButton.disabled =
                true;

            stopButton.disabled =
                true;


            setStatus(
                "Speaking",
                "🔊 Preparing response..."
            );

        }


        // ====================================================
        // BARGE-IN ACK
        // ====================================================

        else if (
            message ===
            "BARGE_IN_ACK"
        ) {

            console.log(
                "✅ Backend received barge-in."
            );

        }


        // ====================================================
        // NO SPEECH
        // ====================================================

        else if (
            message ===
            "NO_SPEECH"
        ) {

            isProcessing =
                false;


            if (
                !isSpeaking &&
                !stoppedByUser
            ) {

                setStatus(
                    "Ready",
                    "⚠️ No speech detected. Try again."
                );


                startButton.disabled =
                    false;

                stopButton.disabled =
                    true;

            }

        }


        // ====================================================
        // ERROR
        // ====================================================

        else if (
            message ===
            "ERROR"
        ) {

            isProcessing =
                false;


            if (
                !isSpeaking
            ) {

                startButton.disabled =
                    false;

                stopButton.disabled =
                    true;


                setStatus(
                    "Error",
                    "❌ Something went wrong."
                );

            }

        }


        // ====================================================
        // TTS ERROR
        // ====================================================

        else if (
            message ===
            "TTS_ERROR"
        ) {

            isSpeaking =
                false;

            isProcessing =
                false;

            stopBargeInDetection();


            setStatus(
                "Error",
                "⚠️ TTS audio could not be generated."
            );


            startButton.disabled =
                false;

            stopButton.disabled =
                true;

        }


        // ====================================================
        // EXIT
        // ====================================================

        else if (
            message ===
            "EXIT"
        ) {

            stoppedByUser =
                true;

            isProcessing =
                false;


            stopAssistantAudio();

            stopRecording();

            closeMicrophone();


            startButton.disabled =
                true;

            stopButton.disabled =
                true;


            setStatus(
                "Disconnected",
                "👋 Voice assistant stopped."
            );

        }

    };


    // ========================================================
    // WEBSOCKET ERROR
    // ========================================================

    socket.onerror =
        function(error) {

        console.error(
            "❌ WebSocket error:",
            error
        );


        isProcessing =
            false;


        stopAssistantAudio();

        stopRecording();


        connectButton.disabled =
            false;

        startButton.disabled =
            true;

        stopButton.disabled =
            true;


        setStatus(
            "Error",
            "❌ Could not connect to WebSocket."
        );

    };


    // ========================================================
    // WEBSOCKET CLOSE
    // ========================================================

    socket.onclose =
        function() {

        console.log(
            "🔌 WebSocket closed."
        );


        stoppedByUser =
            true;


        stopAssistantAudio();

        stopRecording();

        closeMicrophone();


        isProcessing =
            false;


        connectButton.disabled =
            false;

        startButton.disabled =
            true;

        stopButton.disabled =
            true;


        setStatus(
            "Disconnected",
            "🔴 WebSocket disconnected."
        );


        socket =
            null;

    };

}


// ============================================================
// START VOICE
// ============================================================

async function startVoice() {

    if (

        !socket ||

        socket.readyState !==
        WebSocket.OPEN

    ) {

        setStatus(
            "Disconnected",
            "Please click Connect first."
        );

        return;

    }


    if (
        isProcessing ||
        isSpeaking ||
        isRecording
    ) {

        console.log(
            "⚠️ Voice assistant is already active."
        );

        return;

    }


    stoppedByUser =
        false;


    try {

        await ensureMicrophone();

    }

    catch (error) {

        console.error(
            "❌ Microphone error:",
            error
        );

        setStatus(
            "Error",
            "❌ Microphone permission denied."
        );

        return;

    }


    socket.send(
        "start"
    );

}


// ============================================================
// ENSURE MICROPHONE
// ============================================================

async function ensureMicrophone() {

    if (
        audioStream &&
        audioStream.active
    ) {

        return;

    }


    console.log(
        "🎤 Requesting microphone permission..."
    );


    audioStream =
        await navigator.mediaDevices.getUserMedia(
            {
                audio: {

                    echoCancellation: true,

                    noiseSuppression: true,

                    autoGainControl: true,

                    channelCount: 1

                }
            }
        );


    console.log(
        "✅ Microphone stream ready."
    );

}


// ============================================================
// START RECORDING
// ============================================================

async function startRecording() {

    if (
        stoppedByUser
    ) {

        return;

    }


    if (
        isRecording
    ) {

        console.log(
            "⚠️ Recording already active."
        );

        return;

    }


    if (
        isSpeaking
    ) {

        console.log(
            "⚠️ Cannot record while assistant is speaking."
        );

        return;

    }


    try {

        await ensureMicrophone();


        stopNormalSilenceDetection();


        audioChunks = [];

        speechDetected =
            false;

        recordingStartTime =
            Date.now();


        // ====================================================
        // MIME TYPE
        // ====================================================

        let mimeType =
            "audio/webm;codecs=opus";


        if (
            !MediaRecorder.isTypeSupported(
                mimeType
            )
        ) {

            mimeType =
                "audio/webm";

        }


        // ====================================================
        // CREATE RECORDER
        // ====================================================

        mediaRecorder =
            new MediaRecorder(
                audioStream,
                {
                    mimeType:
                        mimeType
                }
            );


        // ====================================================
        // AUDIO DATA
        // ====================================================

        mediaRecorder.ondataavailable =
            function(event) {

            if (

                event.data &&

                event.data.size > 0

            ) {

                audioChunks.push(
                    event.data
                );

            }

        };


        // ====================================================
        // STOP
        // ====================================================

        mediaRecorder.onstop =
            async function() {

            console.log(
                "⏹️ Browser recording stopped."
            );


            isRecording =
                false;


            stopNormalSilenceDetection();


            const audioBlob =
                new Blob(
                    audioChunks,
                    {
                        type:
                            mimeType
                    }
                );


            console.log(
                "🎧 Audio size:",
                audioBlob.size,
                "bytes"
            );


            if (
                audioBlob.size === 0
            ) {

                console.warn(
                    "⚠️ Empty audio."
                );


                if (

                    socket &&

                    socket.readyState ===
                    WebSocket.OPEN

                ) {

                    socket.send(
                        "empty"
                    );

                }

                return;

            }


            // =================================================
            // SEND AUDIO
            // =================================================

            if (

                socket &&

                socket.readyState ===
                WebSocket.OPEN

            ) {

                try {

                    const arrayBuffer =
                        await audioBlob.arrayBuffer();


                    console.log(
                        "📡 Sending audio to FastAPI..."
                    );


                    socket.send(
                        arrayBuffer
                    );


                    console.log(
                        "✅ Audio sent."
                    );

                }

                catch (error) {

                    console.error(
                        "❌ Audio sending error:",
                        error
                    );

                }

            }

        };


        // ====================================================
        // START
        // ====================================================

        mediaRecorder.start(
            100
        );


        isRecording =
            true;


        console.log(
            "🎙️ Browser recording started."
        );


        setStatus(
            "Listening",
            "🎤 Speak now... waiting for silence."
        );


        await startNormalSilenceDetection();

    }

    catch (error) {

        console.error(
            "❌ Recording error:",
            error
        );


        isRecording =
            false;


        setStatus(
            "Error",
            "❌ Could not start microphone recording."
        );

    }

}


// ============================================================
// NORMAL SILENCE DETECTION
// ============================================================

async function startNormalSilenceDetection() {

    try {

        stopNormalSilenceDetection();


        audioContext =
            new (
                window.AudioContext ||
                window.webkitAudioContext
            )();


        if (
            audioContext.state ===
            "suspended"
        ) {

            await audioContext.resume();

        }


        analyser =
            audioContext.createAnalyser();


        analyser.fftSize =
            2048;


        analyser.smoothingTimeConstant =
            0.8;


        microphoneSource =
            audioContext.createMediaStreamSource(
                audioStream
            );


        microphoneSource.connect(
            analyser
        );


        const dataArray =
            new Uint8Array(
                analyser.fftSize
            );


        function checkVolume() {

            if (
                !mediaRecorder ||
                mediaRecorder.state !==
                "recording"
            ) {

                return;

            }


            analyser.getByteTimeDomainData(
                dataArray
            );


            let sum = 0;


            for (
                let i = 0;
                i < dataArray.length;
                i++
            ) {

                const normalized =
                    (
                        dataArray[i] -
                        128
                    ) / 128;


                sum +=
                    normalized *
                    normalized;

            }


            const rms =
                Math.sqrt(
                    sum /
                    dataArray.length
                );


            // =================================================
            // SPEECH
            // =================================================

            if (
                rms >
                SILENCE_THRESHOLD
            ) {

                speechDetected =
                    true;


                if (
                    silenceTimer
                ) {

                    clearTimeout(
                        silenceTimer
                    );

                    silenceTimer =
                        null;

                }

            }

            // =================================================
            // SILENCE
            // =================================================

            else {

                if (

                    speechDetected &&

                    !silenceTimer

                ) {

                    silenceTimer =
                        setTimeout(
                            function() {

                                if (

                                    mediaRecorder &&

                                    mediaRecorder.state ===
                                    "recording"

                                ) {

                                    console.log(
                                        "🛑 Silence timeout."
                                    );


                                    stopRecording();

                                }

                            },
                            SILENCE_DURATION
                        );

                }

            }


            // =================================================
            // MAX TIME
            // =================================================

            if (

                recordingStartTime &&

                Date.now() -
                recordingStartTime >=
                MAX_RECORDING_TIME

            ) {

                console.log(
                    "⏰ Maximum recording time reached."
                );


                stopRecording();

                return;

            }


            volumeCheckTimer =
                requestAnimationFrame(
                    checkVolume
                );

        }


        checkVolume();

    }

    catch (error) {

        console.error(
            "❌ Silence detection error:",
            error
        );

    }

}


// ============================================================
// STOP NORMAL SILENCE DETECTION
// ============================================================

function stopNormalSilenceDetection() {

    if (
        silenceTimer
    ) {

        clearTimeout(
            silenceTimer
        );

        silenceTimer =
            null;

    }


    if (
        volumeCheckTimer
    ) {

        cancelAnimationFrame(
            volumeCheckTimer
        );

        volumeCheckTimer =
            null;

    }


    if (
        microphoneSource
    ) {

        try {

            microphoneSource.disconnect();

        }

        catch (error) {

            console.warn(
                error
            );

        }

        microphoneSource =
            null;

    }


    if (
        audioContext
    ) {

        try {

            audioContext.close();

        }

        catch (error) {

            console.warn(
                error
            );

        }

        audioContext =
            null;

    }


    analyser =
        null;

}


// ============================================================
// STOP RECORDING
// ============================================================

function stopRecording() {

    stopNormalSilenceDetection();


    if (

        mediaRecorder &&

        mediaRecorder.state !==
        "inactive"

    ) {

        try {

            mediaRecorder.stop();

        }

        catch (error) {

            console.warn(
                "Recorder stop warning:",
                error
            );

        }

    }


    mediaRecorder =
        null;

    isRecording =
        false;

}


// ============================================================
// CLOSE MICROPHONE
// ============================================================

function closeMicrophone() {

    stopRecording();

    stopBargeInDetection();


    if (
        audioStream
    ) {

        audioStream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

    }


    audioStream =
        null;

}


// ============================================================
// STOP VOICE
// ============================================================

function stopVoice() {

    console.log(
        "🛑 User pressed Stop."
    );


    stoppedByUser =
        true;


    isProcessing =
        false;


    stopAssistantAudio();

    closeMicrophone();


    if (

        socket &&

        socket.readyState ===
        WebSocket.OPEN

    ) {

        socket.send(
            "stop"
        );

    }


    startButton.disabled =
        false;

    stopButton.disabled =
        true;


    setStatus(
        "Ready",
        "Voice assistant stopped. Click Start Voice to begin again."
    );

}

</script>

</body>

</html>
"""
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "🎙️ DesFlyer Voice Assistant Server"
    )

    print(
        "===================================="
    )

    print(
        "🌐 Open in browser:"
    )

    print(
        "👉 http://127.0.0.1:8000/voice"
    )

    print(
        "\n🔌 WebSocket:"
    )

    print(
        "👉 ws://127.0.0.1:8000/ws/voice"
    )

    print(
        "\n🚀 Starting Uvicorn..."
    )

    print(
        "====================================\n"
    )


    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000,

        reload=False

    )