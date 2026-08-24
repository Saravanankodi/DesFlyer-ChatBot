import re
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad

from scipy.io.wavfile import write
from faster_whisper import WhisperModel


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
FRAME_SIZE = 480                  # 30 ms

# ============================================================
# WEBRTC VAD
# ============================================================

VAD_MODE = 2

vad = webrtcvad.Vad(VAD_MODE)


# ============================================================
# SPEECH DETECTION
# ============================================================

REQUIRED_SPEECH_FRAMES = 10

SILENCE_LIMIT = 40

WAITING_LIMIT = 100

MAX_RECORDING_FRAMES = 500


# ============================================================
# MICROPHONE
# ============================================================

BASE_RMS_THRESHOLD = 500

NOISE_MULTIPLIER = 2.5

MIN_RMS_THRESHOLD = 700

MAX_RMS_THRESHOLD = 3000


# ============================================================
# AUDIO VALIDATION
# ============================================================

MIN_AUDIO_DURATION = 0.7

MIN_VOICED_RATIO = 0.15

MIN_SPEECH_FRAMES = 8

MAX_TEXT_LENGTH = 250


# ============================================================
# WHISPER
# ============================================================

MAX_NO_SPEECH_PROB = 0.70

MAX_COMPRESSION_RATIO = 2.4


# ============================================================
# LOAD WHISPER MODEL
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

    print("✅ Faster-Whisper model loaded successfully.")

except Exception as error:

    print("❌ Could not load Faster-Whisper:")
    print(error)

    raise


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

    if not normalized:
        return False

    words = normalized.split()

    exit_words = {
        "bye",
        "goodbye",
        "exit",
        "quit",
        "stop",
        "close"
    }

    exact_commands = {
        "bye",
        "goodbye",
        "good bye",
        "exit",
        "quit",
        "bye bye",
        "stop",
        "close"
    }

    if normalized in exact_commands:
        return True

    if len(words) <= 6:

        if words[-1] in exit_words:
            return True

    if len(words) >= 2:

        last_two = " ".join(
            words[-2:]
        )

        if last_two in {
            "good bye",
            "bye bye"
        }:
            return True

    return False


# ============================================================
# DESFLYER WORD CORRECTION
# ============================================================

def correct_desflyer_words(text):

    corrections = {

        # Common Whisper mistakes
        "this player": "DesFlyer",
        "desk player": "DesFlyer",
        "desk flyer": "DesFlyer",
        "des flyer": "DesFlyer",
        "this flyer": "DesFlyer",
        "the flyer": "DesFlyer",
        "display": "DesFlyer",
        "des fire": "DesFlyer",
        "des flier": "DesFlyer",
        "desk flier": "DesFlyer",
        "death flyer": "DesFlyer",
        "death player": "DesFlyer",
        "des player": "DesFlyer"

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
# MICROPHONE CALIBRATION
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

        rms = float(
            np.sqrt(
                np.mean(
                    audio.astype(np.float32) ** 2
                )
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

            sd.sleep(1000)

    except Exception as error:

        print(
            "❌ Microphone calibration failed:",
            error
        )

        return BASE_RMS_THRESHOLD

    if not rms_values:

        return BASE_RMS_THRESHOLD

    noise_floor = float(
        np.percentile(
            rms_values,
            80
        )
    )

    dynamic_threshold = max(
        BASE_RMS_THRESHOLD,
        noise_floor * NOISE_MULTIPLIER,
        MIN_RMS_THRESHOLD
    )

    dynamic_threshold = min(
        dynamic_threshold,
        MAX_RMS_THRESHOLD
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
# TRANSCRIPTION VALIDATION
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

    # --------------------------------------------------------
    # Voiced ratio
    # --------------------------------------------------------

    print(
        f"📊 Voiced frame ratio: "
        f"{voiced_ratio:.2f}"
    )

    if voiced_ratio < MIN_VOICED_RATIO:

        print(
            "❌ Too little speech detected."
        )

        return False

    # --------------------------------------------------------
    # No speech probability
    # --------------------------------------------------------

    no_speech_probs = [
        segment.no_speech_prob
        for segment in segments
        if segment.no_speech_prob is not None
    ]

    if no_speech_probs:

        average_no_speech = float(
            np.mean(
                no_speech_probs
            )
        )

        print(
            f"📊 Average no-speech probability: "
            f"{average_no_speech:.2f}"
        )

        if average_no_speech > MAX_NO_SPEECH_PROB:

            print(
                "❌ Audio appears to contain mostly noise."
            )

            return False

    # --------------------------------------------------------
    # Compression ratio
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Empty text
    # --------------------------------------------------------

    if not text.strip():

        print(
            "❌ Empty transcription."
        )

        return False

    return True


# ============================================================
# REMOVE EXCESSIVE WORD REPETITION
# ============================================================

def remove_repeated_words(text):

    words = text.split()

    if not words:
        return text

    cleaned_words = []

    previous_word = None
    repeat_count = 0

    for word in words:

        current_word = word.lower()

        if (
            previous_word is not None
            and current_word == previous_word
        ):

            repeat_count += 1

            if repeat_count >= 2:
                continue

        else:

            repeat_count = 0

        cleaned_words.append(
            word
        )

        previous_word = current_word

    return " ".join(
        cleaned_words
    ).strip()


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text():

    print("\n🎤 Speak now...")
    print("⏳ I am listening. Take your time...")

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
        # Convert audio
        # ----------------------------------------------------

        audio = (
            indata[:, 0] * 32767
        ).astype(np.int16)

        # ----------------------------------------------------
        # RMS
        # ----------------------------------------------------

        rms = float(
            np.sqrt(
                np.mean(
                    audio.astype(np.float32) ** 2
                )
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
        # Combined speech detection
        # ----------------------------------------------------

        real_speech = (
            vad_speech
            and rms >= speech_threshold
        )

        # ====================================================
        # WAITING FOR SPEECH
        # ====================================================

        if not speech_started:

            pre_buffer.append(
                audio.copy()
            )

            if real_speech:

                consecutive_speech_frames += 1

            else:

                consecutive_speech_frames = 0

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

                for frame in pre_buffer:

                    audio_frames.append(
                        frame
                    )

                    speech_flags.append(
                        False
                    )

                pre_buffer.clear()

                silence_count = 0

        # ====================================================
        # RECORDING
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
            # WAIT FOR SPEECH
            # ------------------------------------------------

            waiting_frames = 0

            while not speech_started:

                sd.sleep(100)

                waiting_frames += 1

                if waiting_frames >= WAITING_LIMIT:

                    print(
                        "\n⏳ No speech detected."
                    )

                    return ""

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            while True:

                sd.sleep(100)

                if silence_count >= SILENCE_LIMIT:

                    print(
                        "\n⏹️ User stopped speaking."
                    )

                    break

                if (
                    total_recording_frames
                    >= MAX_RECORDING_FRAMES
                ):

                    print(
                        "\n⏱️ Maximum recording time reached."
                    )

                    break

    except KeyboardInterrupt:

        raise

    except Exception as error:

        print(
            "\n❌ Microphone error:",
            error
        )

        return ""

    # ========================================================
    # VALIDATE AUDIO
    # ========================================================

    if not audio_frames:

        print(
            "❌ No audio recorded."
        )

        return ""

    if actual_speech_frames < MIN_SPEECH_FRAMES:

        print(
            "❌ Not enough speech detected."
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
    # AUDIO INFORMATION
    # ========================================================

    final_rms = float(
        np.sqrt(
            np.mean(
                audio.astype(np.float32) ** 2
            )
        )
    )

    duration = (
        len(audio)
        / SAMPLE_RATE
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

    print(
        f"⏱️ Recording duration: "
        f"{duration:.2f} seconds"
    )

    if duration < MIN_AUDIO_DURATION:

        print(
            "❌ Recording too short."
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
    # WHISPER
    # ========================================================

    print(
        "🔄 Whisper transcribing..."
    )

    try:

        segments, info = stt_model.transcribe(

            "input.wav",

            beam_size=3,

            language="en",

            initial_prompt=(
                "DesFlyer, website development, "
                "mobile applications, software development, "
                "business, services, projects."
            ),

            condition_on_previous_text=False,

            temperature=0.0,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 500
            },

            no_speech_threshold=0.60,

            log_prob_threshold=-1.5,

            compression_ratio_threshold=2.4

        )

        segments = list(
            segments
        )

        text_parts = []

        for segment in segments:

            if (
                segment.no_speech_prob is not None
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
    # VALIDATE TRANSCRIPTION
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

    text = remove_repeated_words(
        text
    )

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
            "❌ Transcription is too long."
        )

        print(
            "📝 Whisper result:",
            text
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
    # EXIT
    # ========================================================

    if is_exit_command(text):

        print(
            "\n👋 Goodbye!"
        )

        return "__EXIT__"

    # ========================================================
    # RETURN ONLY TEXT
    # ========================================================

    return text