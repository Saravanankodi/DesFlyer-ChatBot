import numpy as np
import sounddevice as sd
import webrtcvad
import re

from scipy.io.wavfile import write
from faster_whisper import WhisperModel

from rag import ask_chatbot
from tts import text_to_speech


# ========================================================
# SETTINGS
# ========================================================

SAMPLE_RATE = 16000
FRAME_SIZE = 160


# ========================================================
# WHISPER MODEL
# ========================================================

stt_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)


# ========================================================
# SPEECH TO TEXT
# ========================================================

def speech_to_text():

    print("\n🎤 Speak now...")
    print("⏳ Waiting for speech...")

    # your existing code...
def speech_to_text():

    print("\n🎤 Speak now...")
    print("⏳ Waiting for speech...")

    audio_frames = []

    speech_started = False
    consecutive_speech_frames = 0
    silence_count = 0
    total_recording_frames = 0

    # ========================================================
    # SETTINGS
    # ========================================================
    vad = webrtcvad.Vad(3)
    REQUIRED_SPEECH_FRAMES = 8
    SILENCE_LIMIT = 40
    WAITING_LIMIT = 100
    MAX_RECORDING_FRAMES = 500

    # Minimum audio volume required
    RMS_THRESHOLD = 500

    # ========================================================
    # AUDIO CALLBACK
    # ========================================================

    def audio_callback(
        indata,
        frames_count,
        callback_time,
        status
    ):

        nonlocal speech_started
        nonlocal consecutive_speech_frames
        nonlocal silence_count
        nonlocal total_recording_frames

        if status:
            print("⚠️ Microphone status:", status)

        # Convert microphone audio to int16
        audio = (
            indata[:, 0] * 32767
        ).astype(np.int16)

        # ----------------------------------------------------
        # Calculate audio volume
        # ----------------------------------------------------

        rms = np.sqrt(
            np.mean(
                audio.astype(np.float32) ** 2
            )
        )

        # ----------------------------------------------------
        # WebRTC VAD
        # ----------------------------------------------------

        vad_speech = vad.is_speech(
            audio.tobytes(),
            SAMPLE_RATE
        )

        # Speech is accepted only when:
        # 1. VAD detects speech
        # 2. Audio volume is high enough

        real_speech = (
            vad_speech
            and rms > RMS_THRESHOLD
        )

        # ====================================================
        # BEFORE SPEECH STARTS
        # ====================================================

        if not speech_started:

            if real_speech:

                consecutive_speech_frames += 1

            else:

                consecutive_speech_frames = 0

            # Require continuous speech
            if (
                consecutive_speech_frames
                >= REQUIRED_SPEECH_FRAMES
            ):

                speech_started = True

                print(
                    "🎙️ Real speech detected..."
                )

                audio_frames.append(
                    audio.copy()
                )

                silence_count = 0

        # ====================================================
        # AFTER SPEECH STARTS
        # ====================================================

        else:

            audio_frames.append(
                audio.copy()
            )

            total_recording_frames += 1

            if real_speech:

                silence_count = 0

            else:

                silence_count += 1

    # ========================================================
    # START MICROPHONE
    # ========================================================

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=FRAME_SIZE,
            dtype="float32",
            callback=audio_callback
        ):

            waiting_frames = 0

            # ------------------------------------------------
            # Wait for real speech
            # ------------------------------------------------

            while not speech_started:

                sd.sleep(100)

                waiting_frames += 1

                if waiting_frames >= WAITING_LIMIT:

                    print(
                        "❌ No speech detected."
                    )

                    print(
                        "🎤 Please speak again."
                    )

                    return ""

            # ------------------------------------------------
            # Continue recording
            # ------------------------------------------------

            while True:

                sd.sleep(100)

                # Stop after enough silence
                if silence_count >= SILENCE_LIMIT:

                    break

                # Safety limit
                if (
                    total_recording_frames
                    >= MAX_RECORDING_FRAMES
                ):

                    print(
                        "⏱️ Maximum recording time reached."
                    )

                    break

    except Exception as error:

        print(
            "❌ Microphone error:",
            error
        )

        return ""

    # ========================================================
    # CHECK RECORDED AUDIO
    # ========================================================

    if not audio_frames:

        print(
            "❌ No speech recorded."
        )

        return ""

    print("✅ Speech completed.")
    print("🔄 Converting speech to text...")

    # ========================================================
    # COMBINE AUDIO
    # ========================================================

    audio = np.concatenate(
        audio_frames
    )

    # ========================================================
    # FINAL AUDIO VOLUME CHECK
    # ========================================================

    rms = np.sqrt(
        np.mean(
            audio.astype(np.float32) ** 2
        )
    )

    print(
        f"🔊 Audio volume: {rms:.2f}"
    )

    if rms < RMS_THRESHOLD:

        print(
            "❌ Audio volume too low."
        )

        print(
            "🎤 No clear speech detected. "
            "Please speak again."
        )

        return ""

    # ========================================================
    # SAVE AUDIO
    # ========================================================

    try:

        write(
            "input.wav",
            SAMPLE_RATE,
            audio.astype(np.int16)
        )

        print(
            "💾 Audio saved to input.wav"
        )

    except Exception as error:

        print(
            "❌ Could not save audio:",
            error
        )

        return ""

    # ========================================================
    # WHISPER STT
    # ========================================================

    print(
        "🔄 Whisper transcribing..."
    )

    try:

        segments, info = stt_model.transcribe(

            "input.wav",

            beam_size=5,

            language="en",

            # IMPORTANT:
            # No initial_prompt here.
            # This prevents Whisper from hallucinating
            # DesFlyer-related text when there is silence.

            condition_on_previous_text=False,

            vad_filter=False,

            temperature=0.0,

            no_speech_threshold=0.6,

            log_prob_threshold=-1.0,

            compression_ratio_threshold=2.4
        )

        text = " ".join(
            segment.text
            for segment in segments
        ).strip()

    except Exception as error:

        print(
            "❌ STT transcription error:",
            error
        )

        return ""

    # ========================================================
    # CORRECT COMMON STT ERRORS
    # ========================================================

    corrections = {

        "display": "DesFlyer",

        "des flyer": "DesFlyer",

        "desk flyer": "DesFlyer",

        "this flyer": "DesFlyer",

        "the flyer": "DesFlyer"
    }

    for wrong, correct in corrections.items():

        text = re.sub(
            rf"\b{re.escape(wrong)}\b",
            correct,
            text,
            flags=re.IGNORECASE
        )

    # ========================================================
    # CHECK TRANSCRIPTION
    # ========================================================

    if not text:

        print(
            "❌ I could not understand your speech."
        )

        print(
            "🎤 Please speak again."
        )

        return ""

    print(
        "\n📝 You said:",
        text
    )

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    normalized_text = re.sub(
        r"[^\w\s]",
        "",
        text.lower()
    ).strip()

    # ========================================================
    # EXIT COMMANDS
    # ========================================================

    if normalized_text in [
        "exit",
        "quit",
        "bye"
    ]:

        return normalized_text

    # ========================================================
    # THANK YOU
    # ========================================================

    if normalized_text in [
        "thank you",
        "thanks",
        "thank you very much"
    ]:

        answer = (
            "You're welcome! "
            "Feel free to ask me about DesFlyer."
        )

        print(
            "\n===== Voice Assistant ====="
        )

        print(answer)

        print(
            "🔊 Converting answer to speech..."
        )

        try:

            text_to_speech(answer)

            print(
                "✅ Voice response completed."
            )

        except Exception as error:

            print(
                "❌ TTS error:",
                error
            )

        return text

    # ========================================================
    # SEND TO RAG
    # ========================================================

    print(
        "🤖 Getting answer from RAG..."
    )

    try:

        answer = ask_chatbot(text)

    except Exception as error:

        print(
            "❌ RAG error:",
            error
        )

        return text

    # ========================================================
    # DISPLAY RAG ANSWER
    # ========================================================

    print(
        "\n===== RAG Answer ====="
    )

    print(answer)

    # ========================================================
    # TEXT-TO-SPEECH
    # ========================================================

    if answer:

        print(
            "🔊 Converting answer to speech..."
        )

        try:

            text_to_speech(answer)

            print(
                "✅ Voice response completed."
            )

        except Exception as error:

            print(
                "❌ TTS error:",
                error
            )

    return text