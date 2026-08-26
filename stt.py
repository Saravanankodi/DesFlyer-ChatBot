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

# 30 ms frame
FRAME_SIZE = 480


# ============================================================
# WEBRTC VAD
# ============================================================

# 2 = balanced speech detection
VAD_MODE = 2

vad = webrtcvad.Vad(VAD_MODE)


# ============================================================
# SPEECH DETECTION
# ============================================================

# Reduced from 10 -> 6.
# 6 x 30 ms = 180 ms.
#
# This helps prevent short words at the beginning
# from being missed.
REQUIRED_SPEECH_FRAMES = 6


# 30 x 30 ms = 0.9 seconds.
#
# Reduced from 1.2 sec so short questions can finish
# without unnecessary waiting.
SILENCE_LIMIT = 30


# Waiting time for user to start speaking.
# 100 x 100 ms = approximately 10 seconds.
WAITING_LIMIT = 100


# 500 x 30 ms = approximately 15 seconds.
MAX_RECORDING_FRAMES = 500


# ============================================================
# MICROPHONE
# ============================================================

BASE_RMS_THRESHOLD = 400

NOISE_MULTIPLIER = 2.2

MIN_RMS_THRESHOLD = 550

MAX_RMS_THRESHOLD = 3000


# ============================================================
# AUDIO VALIDATION
# ============================================================

MIN_AUDIO_DURATION = 0.45

MIN_VOICED_RATIO = 0.10

MIN_SPEECH_FRAMES = 5

MAX_TEXT_LENGTH = 250


# ============================================================
# WHISPER VALIDATION
# ============================================================

MAX_NO_SPEECH_PROB = 0.75

MAX_COMPRESSION_RATIO = 3.0


# ============================================================
# WHISPER MODEL
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

    print(
        "✅ Faster-Whisper model loaded successfully."
    )

except Exception as error:

    print(
        "❌ Could not load Faster-Whisper:"
    )

    print(error)

    raise


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not text:
        return ""

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

        if words[-1] in {
            "bye",
            "goodbye",
            "exit",
            "quit",
            "stop",
            "close"
        }:

            return True

    return False


# ============================================================
# DESFLYER WORD CORRECTION
# ============================================================

def correct_desflyer_words(text):

    corrections = {

        # ----------------------------------------------------
        # Common Faster-Whisper variations
        # ----------------------------------------------------

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
        "des player": "DesFlyer",

        # Common variations
        "desflyer": "DesFlyer",
        "des flyers": "DesFlyer",
        "desflyers": "DesFlyer",

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
# DOMAIN WORD NORMALIZATION
# ============================================================

def correct_domain_terms(text):

    replacements = {

        # ----------------------------------------------------
        # Website
        # ----------------------------------------------------

        "web site": "website",
        "web sites": "websites",

        # ----------------------------------------------------
        # Mobile applications
        # ----------------------------------------------------

        "mobile app": "mobile application",
        "mobile apps": "mobile applications",

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        "data base": "database",
        "data bases": "databases",
        "data basis": "databases",

        # ----------------------------------------------------
        # iOS
        # ----------------------------------------------------

        "i os": "iOS",
        "ios": "iOS",

        # ----------------------------------------------------
        # Android
        # ----------------------------------------------------

        "android": "Android",

        # ----------------------------------------------------
        # Responsive
        # ----------------------------------------------------

        "responsive": "responsive",

        # ----------------------------------------------------
        # DesFlyer
        # ----------------------------------------------------

        "des flyer": "DesFlyer",
        "desk flyer": "DesFlyer",
        "desk player": "DesFlyer",

    }

    for wrong, correct in replacements.items():

        text = re.sub(
            rf"\b{re.escape(wrong)}\b",
            correct,
            text,
            flags=re.IGNORECASE
        )

    return text


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

            # Remove only excessive repetition.
            #
            # Example:
            # "web development web development"
            #
            # becomes:
            # "web development"
            if repeat_count >= 2:
                continue

        else:

            repeat_count = 0

        cleaned_words.append(word)

        previous_word = current_word

    return " ".join(
        cleaned_words
    ).strip()


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

        print(
            "⚠️ Could not measure background noise."
        )

        return BASE_RMS_THRESHOLD

    # --------------------------------------------------------
    # Noise floor
    # --------------------------------------------------------

    noise_floor = float(
        np.percentile(
            rms_values,
            80
        )
    )

    # --------------------------------------------------------
    # Dynamic threshold
    # --------------------------------------------------------

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
    # No-speech probability
    # --------------------------------------------------------

    no_speech_probs = [
        segment.no_speech_prob
        for segment in segments
        if segment.no_speech_prob is not None
    ]

    if no_speech_probs:

        average_no_speech = float(
            np.mean(no_speech_probs)
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
# SPEECH TO TEXT
# ============================================================

def speech_to_text():

    print("\n🎤 Speak now...")
    print(
        "⏳ I am listening. Take your time..."
    )

    # --------------------------------------------------------
    # Calibrate microphone
    # --------------------------------------------------------

    speech_threshold = calibrate_microphone()

    audio_frames = []
    speech_flags = []

    speech_started = False

    consecutive_speech_frames = 0
    silence_count = 0

    total_recording_frames = 0
    actual_speech_frames = 0

    # ========================================================
    # PRE-BUFFER
    # ========================================================

    # 20 frames x 30 ms = 600 ms.
    #
    # This is important.
    # It preserves the first part of a sentence before
    # speech detection becomes active.
    pre_buffer = deque(
        maxlen=20
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
        # WebRTC VAD
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

                # Don't immediately reset.
                #
                # This allows small gaps between words.
                if consecutive_speech_frames > 0:

                    consecutive_speech_frames -= 1

            # ------------------------------------------------
            # Confirm speech
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

                # ------------------------------------------------
                # Add pre-buffer.
                #
                # The first words may have started before
                # speech detection was confirmed.
                # ------------------------------------------------

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

                    print(
                        "🎤 Please speak when ready."
                    )

                    return ""

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            while True:

                sd.sleep(100)

                # Stop after approximately
                # 0.9 seconds of silence.
                if silence_count >= SILENCE_LIMIT:

                    print(
                        "\n⏹️ User stopped speaking."
                    )

                    break

                # Maximum recording duration.
                if (
                    total_recording_frames
                    >= MAX_RECORDING_FRAMES
                ):

                    print(
                        "\n⏱️ Maximum recording time reached."
                    )

                    break

    except KeyboardInterrupt:

        print(
            "\n🛑 Recording interrupted."
        )

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
    # WHISPER TRANSCRIPTION
    # ========================================================

    print(
        "🔄 Whisper transcribing..."
    )

    try:

        segments, info = stt_model.transcribe(

            "input.wav",

            # ------------------------------------------------
            # Beam search
            # ------------------------------------------------

            beam_size=5,

            best_of=5,

            language="en",

            # ------------------------------------------------
            # Domain vocabulary
            # ------------------------------------------------

            initial_prompt=(
                "This is a DesFlyer company FAQ conversation. "
                "DesFlyer provides software development, "
                "website development, web development, "
                "mobile application development, "
                "Android and iOS application development, "
                "responsive websites, databases, "
                "custom software solutions, "
                "business solutions, startups, clients, "
                "projects, services, UI and UX. "
                "Important words include: "
                "DesFlyer, website, websites, "
                "web development, software development, "
                "mobile application, mobile applications, "
                "Android, iOS, database, databases, "
                "responsive, scalable, custom, "
                "service, services, client, clients. "
                "Questions may contain short words such as "
                "what, which, who, where, when, why, how, "
                "can, could, do, does, is, are, they, them, "
                "their, it, this, that, these, those."
            ),

            # ------------------------------------------------
            # IMPORTANT
            # ------------------------------------------------
            #
            # Do not let Whisper depend on previous
            # transcription because this can cause it to
            # continue or hallucinate previous words.
            #
            condition_on_previous_text=False,

            temperature=0.0,

            # ------------------------------------------------
            # Whisper VAD disabled here.
            #
            # We already perform speech detection using
            # WebRTC VAD before sending the audio.
            #
            # Applying a second aggressive VAD can remove
            # short words.
            # ------------------------------------------------

            vad_filter=False,

            no_speech_threshold=0.70,

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

            segment_text = segment.text.strip()

            if segment_text:

                text_parts.append(
                    segment_text
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

    # --------------------------------------------------------
    # Remove excessive repeated words
    # --------------------------------------------------------

    text = remove_repeated_words(
        text
    )

    # --------------------------------------------------------
    # DesFlyer correction
    # --------------------------------------------------------

    text = correct_desflyer_words(
        text
    )

    # --------------------------------------------------------
    # Domain corrections
    # --------------------------------------------------------

    text = correct_domain_terms(
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
    # FINAL TEXT
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
    # RETURN TEXT TO APP.PY
    # ========================================================

    return text


# ============================================================
# TEST STT
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("🎙️ STT TEST")
    print("===================================")

    while True:

        text = speech_to_text()

        if text == "__EXIT__":

            print(
                "\n👋 STT test stopped."
            )

            break

        if text:

            print(
                "\n✅ FINAL TRANSCRIPTION:"
            )

            print(text)

        else:

            print(
                "\n⚠️ No valid speech detected."
            )