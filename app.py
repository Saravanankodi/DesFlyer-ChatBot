import asyncio
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
from stt import speech_to_text
from tts import text_to_speech


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="DesFlyer Voice Assistant API",

    description=
        "RAG based Voice Assistant using Gemma 2B",

    version="1.0"
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

            "chat": "/chat",

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

    print(
        "QUESTION:",
        data.question
    )

    print("====================================")


    try:

        start_time = time.time()

        answer = ask_chatbot(
            data.question
        )

        generation_time = (
            time.time()
            - start_time
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

            media_type=
                "text/plain"
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

            media_type=
                "text/plain"
        )


# ============================================================
# WEBSOCKET VOICE ENDPOINT
# ============================================================

@app.websocket("/ws/voice")
async def websocket_voice(
    websocket: WebSocket
):

    await websocket.accept()


    print("\n====================================")
    print("🔌 WebSocket connected")
    print("====================================")


    processing = False


    try:

        # ====================================================
        # CONNECTED
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

            message = (
                await websocket.receive_text()
            )

            message = (
                message.strip().lower()
            )


            print(
                "\n📨 Command:",
                message
            )


            # =================================================
            # START
            # =================================================

            if message in {
                "start",
                "voice",
                "listen"
            }:

                if processing:

                    await websocket.send_text(
                        "BUSY"
                    )

                    continue


                processing = True


                try:

                    # ==========================================
                    # STATE 1
                    # ==========================================

                    await websocket.send_text(
                        "LISTENING"
                    )


                    print(
                        "\n🎤 STATE: LISTENING"
                    )


                    # ==========================================
                    # STT
                    # ==========================================

                    user_text = (
                        await asyncio.to_thread(
                            speech_to_text
                        )
                    )


                    # ==========================================
                    # NO SPEECH
                    # ==========================================

                    if not user_text:

                        await websocket.send_text(
                            "NO_SPEECH"
                        )

                        await websocket.send_text(
                            "READY"
                        )

                        continue


                    # ==========================================
                    # EXIT
                    # ==========================================

                    if user_text == "__EXIT__":

                        await websocket.send_text(
                            "EXIT"
                        )

                        await websocket.send_text(
                            "Goodbye!"
                        )

                        break


                    # ==========================================
                    # STATE 2
                    # PROCESSING
                    # ==========================================

                    await websocket.send_text(
                        "PROCESSING"
                    )


                    print(
                        "\n⚙️ STATE: PROCESSING"
                    )


                    # ==========================================
                    # SEND TRANSCRIPTION
                    # ==========================================

                    await websocket.send_text(

                        "USER:" +
                        user_text
                    )


                    print(
                        "\n🎤 User:",
                        user_text
                    )


                    # ==========================================
                    # RAG
                    # ==========================================

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

                        await websocket.send_text(
                            "ERROR"
                        )

                        await websocket.send_text(
                            "READY"
                        )

                        continue


                    rag_time = (
                        time.time()
                        - rag_start
                    )


                    print(
                        f"⏱️ RAG time: "
                        f"{rag_time:.2f} seconds"
                    )


                    # ==========================================
                    # EMPTY ANSWER
                    # ==========================================

                    if not answer:

                        await websocket.send_text(
                            "ERROR"
                        )

                        await websocket.send_text(
                            "READY"
                        )

                        continue


                    # ==========================================
                    # SEND ANSWER
                    # ==========================================

                    await websocket.send_text(

                        "ANSWER:" +
                        answer
                    )


                    print(
                        "\n🤖 DesFlyer:",
                        answer
                    )


                    # ==========================================
                    # STATE 3
                    # SPEAKING
                    # ==========================================

                    await websocket.send_text(
                        "SPEAKING"
                    )


                    print(
                        "\n🔊 STATE: SPEAKING"
                    )


                    # ==========================================
                    # TTS
                    # ==========================================

                    try:

                        await asyncio.to_thread(

                            text_to_speech,

                            answer
                        )

                        print(
                            "\n✅ TTS completed."
                        )

                    except Exception as error:

                        print(
                            "\n❌ TTS error:",
                            error
                        )


                    # ==========================================
                    # AUDIO SETTLE
                    # ==========================================

                    await asyncio.sleep(
                        0.5
                    )


                    # ==========================================
                    # SPEECH COMPLETED
                    # ==========================================

                    await websocket.send_text(
                        "SPEECH_COMPLETED"
                    )


                    # ==========================================
                    # STATE 4
                    # READY
                    # ==========================================

                    await websocket.send_text(
                        "READY"
                    )


                    print(
                        "\n🔄 STATE: READY"
                    )


                finally:

                    processing = False


            # =================================================
            # STOP
            # =================================================

            elif message in {

                "stop",
                "exit",
                "quit",
                "bye"

            }:

                print(
                    "\n🛑 Stop requested."
                )


                await websocket.send_text(
                    "EXIT"
                )


                await websocket.send_text(
                    "Goodbye!"
                )


                break


            # =================================================
            # PING
            # =================================================

            elif message == "ping":

                await websocket.send_text(
                    "pong"
                )


            # =================================================
            # UNKNOWN
            # =================================================

            else:

                await websocket.send_text(
                    "UNKNOWN_COMMAND"
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

        content="""

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

/* ==========================================================
   GLOBAL
   ========================================================== */

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


/* ==========================================================
   CONTAINER
   ========================================================== */

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


/* ==========================================================
   HEADER
   ========================================================== */

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


/* ==========================================================
   STATUS
   ========================================================== */

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


/* ==========================================================
   STATE COLORS
   ========================================================== */

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


/* ==========================================================
   BUTTONS
   ========================================================== */

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


/* ==========================================================
   PIPELINE
   ========================================================== */

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


/* ==========================================================
   CONVERSATION
   ========================================================== */

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


    <!-- HEADER -->

    <div class="header">

        <h1>
            🎙️ DesFlyer Voice Assistant
        </h1>

        <p>
            RAG-based Voice Assistant
        </p>

    </div>


    <!-- STATUS -->

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


    <!-- BUTTONS -->

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


    <!-- PIPELINE -->

    <div class="pipeline">

        <div class="step">
            🎤 Voice
        </div>

        <div>
            →
        </div>

        <div class="step">
            📝 STT
        </div>

        <div>
            →
        </div>

        <div class="step">
            🔎 RAG
        </div>

        <div>
            →
        </div>

        <div class="step">
            🤖 Gemma
        </div>

        <div>
            →
        </div>

        <div class="step">
            🔊 TTS
        </div>

    </div>


    <!-- CONVERSATION -->

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
// VARIABLES
// ============================================================

let socket = null;

let isProcessing = false;


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
// CLEAR CONVERSATION
// ============================================================

function clearConversation() {

    output.innerHTML = "";

}


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


    const className =
        "status-" +
        state.toLowerCase();


    status.classList.add(
        className
    );


    // --------------------------------------------------------
    // ICON
    // --------------------------------------------------------

    if (
        state === "Disconnected"
    ) {

        icon.innerText =
            "🔴";

    }

    else if (
        state === "Connected"
    ) {

        icon.innerText =
            "🔵";

    }

    else if (
        state === "Ready"
    ) {

        icon.innerText =
            "🟢";

    }

    else if (
        state === "Listening"
    ) {

        icon.innerText =
            "🎤";

    }

    else if (
        state === "Processing"
    ) {

        icon.innerText =
            "⚙️";

    }

    else if (
        state === "Speaking"
    ) {

        icon.innerText =
            "🔊";

    }

    else if (
        state === "Error"
    ) {

        icon.innerText =
            "❌";

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
// CONNECT
// ============================================================

function connectWebSocket() {

    // --------------------------------------------------------
    // Already connected
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // CLEAR OLD CONVERSATION
    // --------------------------------------------------------

    clearConversation();


    setStatus(
        "Connected",
        "Connecting to voice assistant..."
    );


    // --------------------------------------------------------
    // Determine protocol
    // --------------------------------------------------------

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
        "WebSocket URL:",
        wsUrl
    );


    // --------------------------------------------------------
    // Create WebSocket
    // --------------------------------------------------------

    socket =
        new WebSocket(
            wsUrl
        );


    // ========================================================
    // OPEN
    // ========================================================

    socket.onopen = function() {

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
            "Connection established. Preparing assistant..."
        );

    };


    // ========================================================
    // MESSAGE
    // ========================================================

    socket.onmessage = function(event) {

        const message =
            event.data;


        console.log(
            "Server:",
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


            connectButton.disabled =
                true;

            startButton.disabled =
                false;

            stopButton.disabled =
                true;


            setStatus(
                "Ready",
                "Click Start Voice to ask a question."
            );

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
                "🎤 I am listening to your voice..."
            );

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
                "⚙️ Processing your question..."
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
                "🎤 You: " + text,
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
            message === "SPEAKING"
        ) {

            isProcessing =
                true;


            startButton.disabled =
                true;

            stopButton.disabled =
                true;


            setStatus(
                "Speaking",
                "🔊 Speaking the answer..."
            );

        }


        // ====================================================
        // SPEECH COMPLETED
        // ====================================================

        else if (
            message ===
            "SPEECH_COMPLETED"
        ) {

            setStatus(
                "Speaking",
                "🔊 Answer completed."
            );

        }


        // ====================================================
        // NO SPEECH
        // ====================================================

        else if (
            message === "NO_SPEECH"
        ) {

            isProcessing =
                false;


            setStatus(
                "Ready",
                "⚠️ No speech detected. Try again."
            );

        }


        // ====================================================
        // BUSY
        // ====================================================

        else if (
            message === "BUSY"
        ) {

            setStatus(
                "Processing",
                "⏳ Please wait..."
            );

        }


        // ====================================================
        // ERROR
        // ====================================================

        else if (
            message === "ERROR"
        ) {

            isProcessing =
                false;


            startButton.disabled =
                false;

            stopButton.disabled =
                true;


            setStatus(
                "Error",
                "❌ Something went wrong."
            );

        }


        // ====================================================
        // EXIT
        // ====================================================

        else if (
            message === "EXIT"
        ) {

            isProcessing =
                false;


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
    // ERROR
    // ========================================================

    socket.onerror = function(error) {

        console.error(
            "❌ WebSocket error:",
            error
        );


        isProcessing =
            false;


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
    // CLOSE
    // ========================================================

    socket.onclose = function() {

        console.log(
            "🔌 WebSocket closed"
        );


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
            "🔴 WebSocket disconnected. Click Connect to reconnect."
        );


        socket =
            null;

    };

}


// ============================================================
// START VOICE
// ============================================================

function startVoice() {

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


    if (isProcessing) {

        setStatus(
            "Processing",
            "⏳ Please wait for the current request."
        );

        return;

    }


    isProcessing =
        true;


    startButton.disabled =
        true;

    stopButton.disabled =
        false;


    setStatus(
        "Listening",
        "🎤 Starting microphone..."
    );


    socket.send(
        "start"
    );

}


// ============================================================
// STOP
// ============================================================

function stopVoice() {

    if (
        socket &&
        socket.readyState ===
            WebSocket.OPEN
    ) {

        socket.send(
            "stop"
        );

    }

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

    print("\n====================================")
    print("🎙️ DesFlyer Voice Assistant Server")
    print("====================================")

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