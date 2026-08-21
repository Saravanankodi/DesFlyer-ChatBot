import re
import time
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad

from scipy.io.wavfile import write
from faster_whisper import WhisperModel

from rag import ask_chatbot
from tts import text_to_speech


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

# 30 ms frame
FRAME_SIZE = 480

# WebRTC VAD
# 0 = least aggressive
# 3 = most aggressive
VAD_MODE = 2

# ------------------------------------------------------------
# Speech detection
# ------------------------------------------------------------

# 15 × 30 ms = 450 ms
# Speech must continue this long before recording starts.
REQUIRED_SPEECH_FRAMES = 15

# 40 × 30 ms = 1.2 seconds
SILENCE_LIMIT = 40

# Wait around 10 seconds for speech
WAITING_LIMIT = 100

# Maximum recording around 12 seconds
MAX_RECORDING_FRAMES = 400

# Base microphone threshold
BASE_RMS_THRESHOLD = 700

# Background noise multiplier
# Speech must be significantly louder than background noise.
NOISE_MULTIPLIER = 2.5

# Minimum recording duration
MIN_AUDIO_DURATION = 0.8

# Maximum accepted question length
MAX_TEXT_LENGTH = 250

# TTS cooldown
TTS_COOLDOWN = 1.5

# ------------------------------------------------------------
# Additional speech validation
# ------------------------------------------------------------

# Minimum percentage of recorded frames that must contain
# real speech.
MIN_VOICED_RATIO = 0.30

# Minimum number of speech frames after speech starts.
MIN_SPEECH_FRAMES = 12

# Whisper confidence settings
MAX_NO_SPEECH_PROB = 0.55
MIN_AVG_LOGPROB = -1.0
MAX_COMPRESSION_RATIO = 2.0


# ============================================================
# LOAD WHISPER
# ============================================================

print("\n===================================")
print("Loading Speech-to-Text Model")
print("===================================")

try:

    stt_model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    print("✅ STT model loaded successfully.")

except Exception as error:

    print("❌ Could not load STT model:")
    print(error)

    raise


# ============================================================
# WEBRTC VAD
# ============================================================

vad = webrtcvad.Vad(VAD_MODE)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = text.lower().strip()

    text = text.replace("-", " ")

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# EXIT COMMAND
# ============================================================

def is_exit_command(text):

    normalized = normalize_text(text)

    # Keep exit commands simple but require exact match.
    # This prevents sentences such as:
    # "I have to go"
    # from being treated as exit.
    exit_commands = {
        "bye",
        "bye bye",
        "goodbye",
        "good bye",
        "exit",
        "quit"
    }

    return normalized in exit_commands


# ============================================================
# DESFLYER WORD CORRECTIONS
# ============================================================

def correct_desflyer_words(text):

    corrections = {

        "this player": "DesFlyer",
        "desk player": "DesFlyer",
        "desk flyer": "DesFlyer",
        "des flyer": "DesFlyer",
        "this flyer": "DesFlyer",
        "the flyer": "DesFlyer",
        "display": "DesFlyer",
        "des fire": "DesFlyer",
        "des flier": "DesFlyer",
        "desk flier": "DesFlyer"
    }

    for wrong, correct in corrections.items():

        text = re.sub(
            rf"\b{re.escape(wrong)}\b",
            correct,
            text,
            flags=re.IGNORECASE
        )

    return text


# ============================================================
# MICROPHONE NOISE CALIBRATION
# ============================================================

def calibrate_microphone():

    print("\n🎧 Calibrating microphone...")
    print("Please remain silent for 1 second.")

    rms_values = []

    def calibration_callback(
        indata,
        frames_count,
        callback_time,
        status
    ):

        audio = (
            indata[:, 0] * 32767
        ).astype(np.int16)

        rms = np.sqrt(
            np.mean(
                audio.astype(np.float32) ** 2
            )
        )

        rms_values.append(rms)

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=FRAME_SIZE,
            dtype="float32",
            callback=calibration_callback
        ):

            # 1.2 seconds
            sd.sleep(1200)

    except Exception as error:

        print(
            "❌ Microphone calibration failed:",
            error
        )

        return BASE_RMS_THRESHOLD

    if not rms_values:

        print(
            "⚠️ Could not measure background noise."
        )

        return BASE_RMS_THRESHOLD

    # Use median instead of maximum because one sudden noise
    # should not completely destroy the threshold.
    noise_floor = float(
        np.median(rms_values)
    )

    # Dynamic threshold
    dynamic_threshold = max(
        BASE_RMS_THRESHOLD,
        noise_floor * NOISE_MULTIPLIER
    )

    print(
        f"🔊 Background noise level: "
        f"{noise_floor:.2f}"
    )

    print(
        f"🎯 Speech threshold: "
        f"{dynamic_threshold:.2f}"
    )

    return dynamic_threshold


# ============================================================
# CHECK WHISPER RESULT
# ============================================================

def is_reliable_transcription(
    segments,
    text,
    voiced_ratio
):

    if not segments:

        print(
            "❌ Whisper returned no segments."
        )

        return False


    # ========================================================
    # 1. VOICED RATIO
    # ========================================================

    print(
        f"📊 Voiced frame ratio: "
        f"{voiced_ratio:.2f}"
    )

    if voiced_ratio < MIN_VOICED_RATIO:

        print(
            "❌ Too little actual speech detected."
        )

        return False


    # ========================================================
    # 2. NO SPEECH PROBABILITY
    # ========================================================

    no_speech_probs = [
        segment.no_speech_prob
        for segment in segments
        if segment.no_speech_prob is not None
    ]

    if no_speech_probs:

        average_no_speech = np.mean(
            no_speech_probs
        )

        print(
            f"📊 Average no-speech probability: "
            f"{average_no_speech:.2f}"
        )

        if average_no_speech > MAX_NO_SPEECH_PROB:

            print(
                "❌ Whisper thinks this is mostly silence/noise."
            )

            return False


    # ========================================================
    # 3. LOG PROBABILITY
    # ========================================================

    log_probs = [
        segment.avg_logprob
        for segment in segments
        if segment.avg_logprob is not None
    ]

    if log_probs:

        average_logprob = np.mean(
            log_probs
        )

        print(
            f"📊 Average log probability: "
            f"{average_logprob:.2f}"
        )

        if average_logprob < MIN_AVG_LOGPROB:

            print(
                "❌ Whisper confidence is too low."
            )

            return False


    # ========================================================
    # 4. COMPRESSION RATIO
    # ========================================================

    compression_ratios = [
        segment.compression_ratio
        for segment in segments
        if segment.compression_ratio is not None
    ]

    if compression_ratios:

        max_compression = max(
            compression_ratios
        )

        print(
            f"📊 Compression ratio: "
            f"{max_compression:.2f}"
        )

        if max_compression > MAX_COMPRESSION_RATIO:

            print(
                "❌ Possible Whisper hallucination."
            )

            return False


    # ========================================================
    # 5. EMPTY TEXT
    # ========================================================

    if not text.strip():

        print(
            "❌ Empty transcription."
        )

        return False


    # ========================================================
    # 6. REPETITION
    # ========================================================

    words = text.lower().split()

    if len(words) >= 6:

        counts = {}

        for word in words:

            counts[word] = (
                counts.get(word, 0) + 1
            )

        highest_count = max(
            counts.values()
        )

        repetition_ratio = (
            highest_count / len(words)
        )

        print(
            f"📊 Repetition ratio: "
            f"{repetition_ratio:.2f}"
        )

        if repetition_ratio > 0.55:

            print(
                "❌ Repetitive transcription detected."
            )

            return False


    # ========================================================
    # 7. REPEATED PHRASES
    # ========================================================

    words = [
        word.lower()
        for word in words
    ]

    if len(words) >= 8:

        for i in range(
            len(words) - 3
        ):

            phrase = words[i:i + 3]

            later = words[i + 3:]

            for j in range(
                len(later) - 2
            ):

                if later[j:j + 3] == phrase:

                    print(
                        "❌ Repeated phrase detected."
                    )

                    return False


    return True


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text():

    print("\n🎤 Speak now...")
    print("⏳ I am listening. Take your time...")

    # --------------------------------------------------------
    # Calibrate microphone before each listening session.
    # --------------------------------------------------------

    speech_threshold = calibrate_microphone()

    audio_frames = []

    speech_flags = []

    speech_started = False

    consecutive_speech_frames = 0

    silence_count = 0

    total_recording_frames = 0

    actual_speech_frames = 0

    pre_buffer = deque(
        maxlen=10
    )


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
        nonlocal actual_speech_frames

        if status:

            print(
                "⚠️ Microphone:",
                status
            )


        # ----------------------------------------------------
        # Convert to int16
        # ----------------------------------------------------

        audio = (
            indata[:, 0] * 32767
        ).astype(np.int16)


        # ----------------------------------------------------
        # RMS
        # ----------------------------------------------------

        rms = np.sqrt(
            np.mean(
                audio.astype(np.float32) ** 2
            )
        )


        # ----------------------------------------------------
        # VAD
        # ----------------------------------------------------

        try:

            vad_speech = vad.is_speech(
                audio.tobytes(),
                SAMPLE_RATE
            )

        except Exception:

            vad_speech = False


        # ----------------------------------------------------
        # Real speech
        # ----------------------------------------------------

        real_speech = (
            vad_speech
            and rms >= speech_threshold
        )


        # ====================================================
        # BEFORE SPEECH
        # ====================================================

        if not speech_started:

            pre_buffer.append(
                audio.copy()
            )

            if real_speech:

                consecutive_speech_frames += 1

            else:

                consecutive_speech_frames = 0


            # ------------------------------------------------
            # Require continuous speech
            # ------------------------------------------------

            if (
                consecutive_speech_frames
                >= REQUIRED_SPEECH_FRAMES
            ):

                speech_started = True

                print(
                    "\n🎙️ Speech detected."
                )

                print(
                    "👂 Listening..."
                )

                # Add previous audio
                for frame in pre_buffer:

                    audio_frames.append(
                        frame
                    )

                    speech_flags.append(
                        False
                    )

                pre_buffer.clear()

                silence_count = 0

                actual_speech_frames = 0


        # ====================================================
        # AFTER SPEECH
        # ====================================================

        else:

            audio_frames.append(
                audio.copy()
            )

            speech_flags.append(
                real_speech
            )

            total_recording_frames += 1


            if real_speech:

                actual_speech_frames += 1

                silence_count = 0

            else:

                silence_count += 1


    # ========================================================
    # OPEN MICROPHONE
    # ========================================================

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=FRAME_SIZE,
            dtype="float32",
            callback=audio_callback
        ):

            # ------------------------------------------------
            # Wait for speech
            # ------------------------------------------------

            waiting_frames = 0

            while not speech_started:

                sd.sleep(100)

                waiting_frames += 1

                if (
                    waiting_frames
                    >= WAITING_LIMIT
                ):

                    print(
                        "\n⏳ No speech detected."
                    )

                    print(
                        "🎤 Please speak when you are ready."
                    )

                    return ""


            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            while True:

                sd.sleep(100)


                # --------------------------------------------
                # Stop after silence
                # --------------------------------------------

                if (
                    silence_count
                    >= SILENCE_LIMIT
                ):

                    print(
                        "\n⏹️ User stopped speaking."
                    )

                    break


                # --------------------------------------------
                # Maximum recording
                # --------------------------------------------

                if (
                    total_recording_frames
                    >= MAX_RECORDING_FRAMES
                ):

                    print(
                        "\n⏱️ Maximum recording time reached."
                    )

                    break


    except Exception as error:

        print(
            "\n❌ Microphone error:",
            error
        )

        return ""


    # ========================================================
    # CHECK AUDIO
    # ========================================================

    if not audio_frames:

        print(
            "❌ No audio recorded."
        )

        return ""


    # ========================================================
    # CHECK MINIMUM SPEECH FRAMES
    # ========================================================

    if (
        actual_speech_frames
        < MIN_SPEECH_FRAMES
    ):

        print(
            "❌ Not enough actual speech detected."
        )

        print(
            "🎤 Please speak clearly."
        )

        return ""


    # ========================================================
    # COMBINE AUDIO
    # ========================================================

    audio = np.concatenate(
        audio_frames
    )


    # ========================================================
    # VOICED RATIO
    # ========================================================

    if speech_flags:

        voiced_ratio = (
            sum(speech_flags)
            / len(speech_flags)
        )

    else:

        voiced_ratio = 0.0


    # ========================================================
    # FINAL RMS
    # ========================================================

    final_rms = np.sqrt(
        np.mean(
            audio.astype(np.float32) ** 2
        )
    )

    print(
        f"🔊 Final audio level: "
        f"{final_rms:.2f}"
    )


    print(
        f"📊 Actual speech frames: "
        f"{actual_speech_frames}"
    )

    print(
        f"📊 Voiced ratio: "
        f"{voiced_ratio:.2f}"
    )


    # ========================================================
    # FINAL VOLUME CHECK
    # ========================================================

    if final_rms < speech_threshold:

        print(
            "❌ Audio too quiet."
        )

        print(
            "🎤 No clear speech detected."
        )

        return ""


    # ========================================================
    # DURATION
    # ========================================================

    duration = (
        len(audio)
        / SAMPLE_RATE
    )

    print(
        f"⏱️ Recording duration: "
        f"{duration:.2f} seconds"
    )


    if duration < MIN_AUDIO_DURATION:

        print(
            "❌ Recording too short."
        )

        print(
            "🎤 Please speak a complete question."
        )

        return ""


    # ========================================================
    # SAVE WAV
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
    # WHISPER
    # ========================================================

    print(
        "🔄 Whisper transcribing..."
    )

    try:

        segments, info = stt_model.transcribe(

            "input.wav",

            beam_size=5,

            language="en",

            condition_on_previous_text=False,

            temperature=0.0,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 700
            },

            no_speech_threshold=0.60,

            log_prob_threshold=-1.0,

            compression_ratio_threshold=2.0

        )

        segments = list(segments)


        # ----------------------------------------------------
        # Build transcription
        # ----------------------------------------------------

        text_parts = []

        for segment in segments:

            if (
                segment.no_speech_prob
                is not None
                and segment.no_speech_prob
                > MAX_NO_SPEECH_PROB
            ):

                continue

            if segment.text.strip():

                text_parts.append(
                    segment.text.strip()
                )


        text = " ".join(
            text_parts
        ).strip()


    except Exception as error:

        print(
            "❌ Whisper error:",
            error
        )

        return ""


    # ========================================================
    # RELIABILITY CHECK
    # ========================================================

    if not is_reliable_transcription(
        segments,
        text,
        voiced_ratio
    ):

        print(
            "🎤 Please speak again."
        )

        return ""


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    if not text:

        print(
            "❌ No understandable speech."
        )

        return ""


    # ========================================================
    # REMOVE EXACT REPEATED WORDS
    # ========================================================

    words = text.split()

    cleaned_words = []

    repeat_count = 0

    previous_word = None


    for word in words:

        if (
            previous_word
            and word.lower()
            == previous_word.lower()
        ):

            repeat_count += 1

            if repeat_count >= 2:

                continue

        else:

            repeat_count = 0


        cleaned_words.append(
            word
        )

        previous_word = word


    text = " ".join(
        cleaned_words
    ).strip()


    # ========================================================
    # DESFLYER CORRECTION
    # ========================================================

    text = correct_desflyer_words(
        text
    )


    # ========================================================
    # LENGTH CHECK
    # ========================================================

    if len(text) > MAX_TEXT_LENGTH:

        print(
            "❌ Transcription is too long/unclear."
        )

        print(
            "📝 Whisper result:",
            text
        )

        print(
            "🎤 Please ask a shorter question."
        )

        return ""


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n📝 You said:"
    )

    print(
        "   ",
        text
    )


    # ========================================================
    # EXIT COMMAND
    # ========================================================

    if is_exit_command(text):

        print(
            "\n👋 Goodbye!"
        )

        return "__EXIT__"


    # ========================================================
    # THANK YOU
    # ========================================================

    normalized = normalize_text(
        text
    )


    if normalized in {
        "thank you",
        "thanks",
        "thank you very much",
        "thanks very much"
    }:

        answer = (
            "You're welcome! "
            "Feel free to ask me about DesFlyer."
        )

        print(
            "\n===== Voice Assistant ====="
        )

        print(
            answer
        )

        print(
            "\n🔊 Speaking answer..."
        )

        try:

            text_to_speech(
                answer
            )

        except Exception as error:

            print(
                "❌ TTS error:",
                error
            )

        time.sleep(
            TTS_COOLDOWN
        )

        print(
            "🎤 Ready for your next question."
        )

        return text


    # ========================================================
    # RAG
    # ========================================================

    print(
        "\n🤖 Getting answer from RAG..."
    )

    try:

        answer = ask_chatbot(
            text
        )

    except Exception as error:

        print(
            "❌ RAG error:",
            error
        )

        return text


    # ========================================================
    # RAG ANSWER
    # ========================================================

    if not answer:

        print(
            "❌ No answer generated."
        )

        return text


    print(
        "\n===== RAG Answer ====="
    )

    print(
        answer
    )


    # ========================================================
    # TTS
    # ========================================================

    print(
        "\n🔊 Speaking answer..."
    )

    try:

        text_to_speech(
            answer
        )

    except Exception as error:

        print(
            "❌ TTS error:",
            error
        )


    # --------------------------------------------------------
    # Allow microphone/speaker to settle
    # --------------------------------------------------------

    time.sleep(
        TTS_COOLDOWN
    )

    print(
        "\n🎤 Ready for your next question."
    )

    return text


# ============================================================
# MAIN
# ============================================================

def main():

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
        "Ask your question naturally."
    )

    print(
        "You can pause while speaking."
    )

    print(
        "Say 'bye', 'goodbye', 'exit' or 'quit' to stop."
    )

    print(
        "===================================="
    )


    while True:

        result = speech_to_text()


        # ----------------------------------------------------
        # No reliable speech
        # ----------------------------------------------------

        if not result:

            continue


        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if result == "__EXIT__":

            print(
                "\n👋 DesFlyer Voice Assistant stopped."
            )

            break


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()