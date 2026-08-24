import asyncio
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from rag import ask_chatbot
from stt import speech_to_text
from tts import text_to_speech


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DesFlyer Voice Assistant API",
    description="RAG based Voice Assistant using Gemma 2B",
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
        "message": "DesFlyer Voice Assistant API is running",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "voice": "/voice",
            "websocket": "/ws/voice"
        }
    }


# ============================================================
# NORMAL CHAT
# ============================================================

@app.post("/chat")
def chat(data: Question):

    print("\n====================================")
    print("🎤 User:", data.question)
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
            "question": data.question,
            "answer": answer
        }

    except Exception as error:

        print(
            "\n❌ Chat error:",
            error
        )

        return {
            "question": data.question,
            "answer": "Sorry, an error occurred while generating the answer."
        }


# ============================================================
# STREAMING CHAT
# ============================================================
#
# NOTE:
# Your current ask_chatbot() generates the complete answer
# before returning it.
#
# Therefore this endpoint currently sends the completed answer
# as one response rather than token-by-token streaming.
#
# ============================================================

@app.post("/chat/stream")
def chat_stream(data: Question):

    print("\n====================================")
    print("STREAMING QUESTION:", data.question)
    print("====================================")

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
            "\n❌ Streaming error:",
            error
        )

        return StreamingResponse(
            iter([
                "Sorry, an error occurred."
            ]),
            media_type="text/plain"
        )


# ============================================================
# PROCESS ONE VOICE QUESTION
# ============================================================
#
# Pipeline:
#
# Microphone
#     ↓
# VAD
#     ↓
# Faster-Whisper
#     ↓
# Text
#     ↓
# RAG
#     ↓
# Gemma
#     ↓
# TTS
#
# speech_to_text() handles microphone + VAD + Whisper.
#
# app.py handles RAG + TTS + WebSocket communication.
#
# ============================================================

def process_voice_question():

    # ========================================================
    # STEP 1 - SPEECH TO TEXT
    # ========================================================

    print("\n🎤 Waiting for your question...")

    try:

        text = speech_to_text()

    except KeyboardInterrupt:

        raise

    except Exception as error:

        print(
            "\n❌ STT error:",
            error
        )

        return {
            "status": "error",
            "text": "",
            "answer": ""
        }


    # ========================================================
    # NO SPEECH
    # ========================================================

    if not text:

        print(
            "\n⚠️ No valid speech detected."
        )

        return {
            "status": "empty",
            "text": "",
            "answer": ""
        }


    # ========================================================
    # EXIT
    # ========================================================

    if text == "__EXIT__":

        print(
            "\n👋 Exit command received."
        )

        return {
            "status": "exit",
            "text": text,
            "answer": ""
        }


    # ========================================================
    # STEP 2 - RAG
    # ========================================================

    print(
        "\n🤖 Getting answer from RAG..."
    )

    start_time = time.time()

    try:

        answer = ask_chatbot(
            text
        )

    except Exception as error:

        print(
            "\n❌ RAG error:"
        )

        print(error)

        return {
            "status": "error",
            "text": text,
            "answer": ""
        }


    generation_time = (
        time.time() - start_time
    )


    # ========================================================
    # CHECK ANSWER
    # ========================================================

    if not answer:

        print(
            "\n❌ RAG returned an empty answer."
        )

        return {
            "status": "error",
            "text": text,
            "answer": ""
        }


    print(
        f"\n⏱️ Generation time: "
        f"{generation_time:.2f} seconds"
    )


    # ========================================================
    # STEP 3 - TTS
    # ========================================================

    print(
        "\n🔊 Speaking answer..."
    )

    try:

        text_to_speech(
            answer
        )

        print(
            "✅ Speech completed."
        )

    except Exception as error:

        print(
            "\n❌ TTS error:"
        )

        print(error)

        # TTS failure should not destroy
        # the WebSocket conversation.


    # ========================================================
    # SMALL MICROPHONE/SPEAKER SETTLE TIME
    # ========================================================

    time.sleep(0.5)


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "status": "success",
        "text": text,
        "answer": answer
    }


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


    # --------------------------------------------------------
    # Prevent multiple voice processing operations
    # at the same time.
    # --------------------------------------------------------

    processing = False


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
        # KEEP CONNECTION ALIVE
        # ====================================================

        while True:

            # ------------------------------------------------
            # Wait for browser command
            # ------------------------------------------------

            message = await websocket.receive_text()

            message = message.strip().lower()


            print(
                "\n📨 WebSocket command:",
                message
            )


            # =================================================
            # START VOICE
            # =================================================

            if message in {
                "start",
                "voice",
                "listen"
            }:


                # ------------------------------------------------
                # Prevent duplicate start commands
                # ------------------------------------------------

                if processing:

                    await websocket.send_text(
                        "BUSY"
                    )

                    continue


                processing = True


                try:

                    # --------------------------------------------
                    # Tell browser we are listening
                    # --------------------------------------------

                    await websocket.send_text(
                        "LISTENING"
                    )


                    print(
                        "\n🎤 Starting voice input..."
                    )


                    # --------------------------------------------
                    # Run blocking operation in thread
                    # --------------------------------------------

                    result = await asyncio.to_thread(
                        process_voice_question
                    )


                    # =================================================
                    # EXIT
                    # =================================================

                    if result["status"] == "exit":

                        await websocket.send_text(
                            "EXIT"
                        )

                        await websocket.send_text(
                            "Goodbye!"
                        )

                        break


                    # =================================================
                    # NO SPEECH
                    # =================================================

                    if result["status"] == "empty":

                        await websocket.send_text(
                            "NO_SPEECH"
                        )

                        await websocket.send_text(
                            "READY"
                        )

                        continue


                    # =================================================
                    # ERROR
                    # =================================================

                    if result["status"] == "error":

                        await websocket.send_text(
                            "ERROR"
                        )

                        await websocket.send_text(
                            "READY"
                        )

                        continue


                    # =================================================
                    # SUCCESS
                    # =================================================

                    if result["status"] == "success":

                        user_text = result["text"]

                        answer = result["answer"]


                        # --------------------------------------------
                        # PRINT USER QUESTION ONLY ONCE
                        # --------------------------------------------

                        print(
                            "\n🎤 User:",
                            user_text
                        )


                        # --------------------------------------------
                        # PRINT ANSWER ONLY ONCE
                        # --------------------------------------------

                        print(
                            "\n🤖 DesFlyer:",
                            answer
                        )


                        # --------------------------------------------
                        # SEND USER QUESTION
                        # --------------------------------------------

                        await websocket.send_text(
                            "USER:" + user_text
                        )


                        # --------------------------------------------
                        # SEND CHATBOT ANSWER
                        # --------------------------------------------

                        await websocket.send_text(
                            "ANSWER:" + answer
                        )


                        # --------------------------------------------
                        # TELL FRONTEND ANSWER IS COMPLETE
                        # --------------------------------------------

                        await websocket.send_text(
                            "SPEECH_COMPLETED"
                        )


                        # --------------------------------------------
                        # SAME CONNECTION READY AGAIN
                        # --------------------------------------------

                        await websocket.send_text(
                            "READY"
                        )


                        print(
                            "\n🔄 Ready for next question."
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
                    "\n🛑 Stop command received."
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
            # UNKNOWN COMMAND
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
            "\n❌ WebSocket error:"
        )

        print(error)


    finally:

        print(
            "🔌 WebSocket connection closed."
        )


# ============================================================
# BROWSER WEBSOCKET TEST PAGE
# ============================================================

@app.get(
    "/voice",
    response_class=HTMLResponse
)
def voice_page():

    html = """
<!DOCTYPE html>

<html>

<head>

    <title>DesFlyer Voice Assistant</title>

    <style>

        body {

            font-family: Arial, sans-serif;

            background: #111;

            color: white;

            text-align: center;

            padding-top: 60px;

        }


        button {

            padding: 14px 28px;

            margin: 10px;

            font-size: 18px;

            cursor: pointer;

        }


        #status {

            margin-top: 30px;

            font-size: 20px;

        }


        #output {

            margin: 30px auto;

            width: 70%;

            min-height: 100px;

            max-height: 400px;

            overflow-y: auto;

            padding: 20px;

            background: #222;

            border-radius: 10px;

            text-align: left;

        }


        .user {

            margin-bottom: 15px;

        }


        .assistant {

            margin-bottom: 20px;

        }

    </style>

</head>


<body>


    <h1>
        🎙️ DesFlyer Voice Assistant
    </h1>


    <p>
        WebSocket Voice Communication
    </p>


    <button onclick="connectWebSocket()">
        🔌 Connect
    </button>


    <button onclick="startVoice()">
        🎤 Start Voice
    </button>


    <button onclick="stopVoice()">
        🛑 Stop
    </button>


    <div id="status">
        Disconnected
    </div>


    <div id="output">

        <b>Conversation:</b>

        <br><br>

    </div>


<script>


let socket = null;

let isProcessing = false;


// ============================================================
// STATUS
// ============================================================

function setStatus(message) {

    document.getElementById(
        "status"
    ).innerText = message;

}


// ============================================================
// OUTPUT
// ============================================================

function addOutput(
    message,
    className = ""
) {

    const output =
        document.getElementById(
            "output"
        );


    const line =
        document.createElement(
            "div"
        );


    line.className =
        className;


    line.innerText =
        message;


    output.appendChild(
        line
    );


    output.scrollTop =
        output.scrollHeight;

}


// ============================================================
// CONNECT WEBSOCKET
// ============================================================

function connectWebSocket() {


    // --------------------------------------------------------
    // Already connected
    // --------------------------------------------------------

    if (
        socket &&
        socket.readyState === WebSocket.OPEN
    ) {

        setStatus(
            "Already connected"
        );

        return;

    }


    // --------------------------------------------------------
    // WebSocket protocol
    // --------------------------------------------------------

    const protocol =
        window.location.protocol === "https:"
        ? "wss:"
        : "ws:";


    const wsUrl =
        protocol +
        "//" +
        window.location.host +
        "/ws/voice";


    console.log(
        "Connecting to:",
        wsUrl
    );


    // --------------------------------------------------------
    // Create connection
    // --------------------------------------------------------

    socket =
        new WebSocket(
            wsUrl
        );


    // ========================================================
    // OPEN
    // ========================================================

    socket.onopen = function() {

        setStatus(
            "🟢 WebSocket Connected"
        );


        addOutput(
            "🔌 WebSocket connected"
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


        // ----------------------------------------------------
        // CONNECTED
        // ----------------------------------------------------

        if (
            message === "CONNECTED"
        ) {

            setStatus(
                "🟢 Connected"
            );

        }


        // ----------------------------------------------------
        // READY
        // ----------------------------------------------------

        else if (
            message === "READY"
        ) {

            isProcessing = false;


            setStatus(
                "🟢 Ready - Click Start Voice"
            );

        }


        // ----------------------------------------------------
        // LISTENING
        // ----------------------------------------------------

        else if (
            message === "LISTENING"
        ) {

            isProcessing = true;


            setStatus(
                "🎤 Listening..."
            );

        }


        // ----------------------------------------------------
        // BUSY
        // ----------------------------------------------------

        else if (
            message === "BUSY"
        ) {

            setStatus(
                "⏳ Please wait..."
            );

        }


        // ----------------------------------------------------
        // NO SPEECH
        // ----------------------------------------------------

        else if (
            message === "NO_SPEECH"
        ) {

            isProcessing = false;


            setStatus(
                "⚠️ No speech detected"
            );

        }


        // ----------------------------------------------------
        // USER QUESTION
        // ----------------------------------------------------

        else if (
            message.startsWith(
                "USER:"
            )
        ) {

            const text =
                message.substring(
                    5
                );


            addOutput(
                "🎤 You: " + text,
                "user"
            );

        }


        // ----------------------------------------------------
        // ANSWER
        // ----------------------------------------------------

        else if (
            message.startsWith(
                "ANSWER:"
            )
        ) {

            const answer =
                message.substring(
                    7
                );


            addOutput(
                "🤖 DesFlyer: " + answer,
                "assistant"
            );


            setStatus(
                "🔊 Speaking answer..."
            );

        }


        // ----------------------------------------------------
        // SPEECH COMPLETED
        // ----------------------------------------------------

        else if (
            message === "SPEECH_COMPLETED"
        ) {

            setStatus(
                "🔊 Answer completed"
            );

        }


        // ----------------------------------------------------
        // ERROR
        // ----------------------------------------------------

        else if (
            message === "ERROR"
        ) {

            isProcessing = false;


            setStatus(
                "❌ Error"
            );

        }


        // ----------------------------------------------------
        // EXIT
        // ----------------------------------------------------

        else if (
            message === "EXIT"
        ) {

            isProcessing = false;


            setStatus(
                "👋 Assistant stopped"
            );

        }

    };


    // ========================================================
    // ERROR
    // ========================================================

    socket.onerror = function(error) {

        console.error(
            "WebSocket error:",
            error
        );


        setStatus(
            "❌ WebSocket error"
        );

    };


    // ========================================================
    // CLOSE
    // ========================================================

    socket.onclose = function() {

        isProcessing = false;


        setStatus(
            "🔴 WebSocket disconnected"
        );


        addOutput(
            "🔌 WebSocket disconnected"
        );


        socket = null;

    };

}


// ============================================================
// START VOICE
// ============================================================

function startVoice() {


    if (
        !socket ||
        socket.readyState !== WebSocket.OPEN
    ) {

        alert(
            "Please connect WebSocket first."
        );

        return;

    }


    if (isProcessing) {

        setStatus(
            "⏳ Please wait for the current answer."
        );

        return;

    }


    isProcessing = true;


    setStatus(
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
        socket.readyState === WebSocket.OPEN
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


    return HTMLResponse(
        content=html
    )


# ============================================================
# COMMAND LINE VOICE ASSISTANT
# ============================================================

def voice_assistant():

    print(
        "\n===================================="
    )

    print(
        "🎙️ DesFlyer Voice Assistant"
    )

    print(
        "===================================="
    )

    print(
        "🎤 Speak naturally."
    )

    print(
        "🤖 I will listen → understand → answer."
    )

    print(
        "🔊 The answer will be spoken."
    )

    print(
        "🛑 Say 'bye', 'exit' or 'quit' to stop."
    )

    print(
        "===================================="
    )


    try:

        while True:

            print(
                "\n🎤 Waiting for your question..."
            )


            result = process_voice_question()


            # =================================================
            # EMPTY
            # =================================================

            if result["status"] == "empty":

                continue


            # =================================================
            # ERROR
            # =================================================

            if result["status"] == "error":

                continue


            # =================================================
            # EXIT
            # =================================================

            if result["status"] == "exit":

                print(
                    "\n👋 DesFlyer Voice Assistant stopped."
                )

                break


            # =================================================
            # SUCCESS
            # =================================================

            if result["status"] == "success":

                print(
                    "\n🎤 Ready for next question."
                )

                time.sleep(
                    0.5
                )


    except KeyboardInterrupt:

        print(
            "\n\n🛑 Assistant stopped by user."
        )


    except Exception as error:

        print(
            "\n❌ Unexpected error:"
        )

        print(error)


    finally:

        print(
            "\n===================================="
        )

        print(
            "🎙️ Voice Assistant Closed"
        )

        print(
            "===================================="
        )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    voice_assistant()