import os
import re
import tempfile

import numpy as np
from faster_whisper import WhisperModel


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

MAX_TEXT_LENGTH = 250

MIN_AUDIO_DURATION = 0.45

MIN_VOICED_RATIO = 0.05

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

        "web site": "website",
        "web sites": "websites",

        "mobile app": "mobile application",
        "mobile apps": "mobile applications",

        "data base": "database",
        "data bases": "databases",
        "data basis": "databases",

        "i os": "iOS",
        "ios": "iOS",

        "android": "Android",

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
# TRANSCRIPTION VALIDATION
# ============================================================

def is_reliable_transcription(
    segments,
    text
):

    if not segments:

        print(
            "❌ Whisper returned no segments."
        )

        return False


    # ========================================================
    # NO-SPEECH PROBABILITY
    # ========================================================

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


    # ========================================================
    # COMPRESSION RATIO
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
    # EMPTY TEXT
    # ========================================================

    if not text.strip():

        print(
            "❌ Empty transcription."
        )

        return False


    return True


# ============================================================
# TRANSCRIBE AUDIO FILE
# ============================================================

def transcribe_audio_file(
    audio_path
):

    if not audio_path:

        print(
            "❌ No audio file received."
        )

        return ""


    if not os.path.exists(audio_path):

        print(
            "❌ Audio file does not exist."
        )

        return ""


    print(
        "\n🔄 Whisper transcribing..."
    )


    try:

        segments, info = stt_model.transcribe(

            audio_path,

            # ------------------------------------------------
            # Beam search
            # ------------------------------------------------

            beam_size=5,

            best_of=5,

            # ------------------------------------------------
            # Language
            #
            # Automatically detect Tamil or English.
            # Also supports multilingual/code-switched
            # speech such as Tamil + English.
            # ------------------------------------------------

            language=None,

            # ------------------------------------------------
            # DesFlyer domain vocabulary
            # ------------------------------------------------

            initial_prompt=(

                "This is a DesFlyer company FAQ conversation. "

                "The user may speak in English, Tamil, "
                "or mixed Tamil-English. "

                "DesFlyer provides software development, "
                "website development, web development, "
                "mobile application development, "
                "Android and iOS application development, "
                "responsive websites, databases, "
                "custom software solutions, "
                "business solutions, startups, clients, "
                "projects, services, UI and UX. "

                "Important English words include: "

                "DesFlyer, website, websites, "
                "web development, software development, "
                "mobile application, mobile applications, "
                "Android, iOS, database, databases, "
                "responsive, scalable, custom, "
                "service, services, client, clients. "

                "Tamil and mixed-language examples may include "
                "questions such as: "

                "DesFlyer enna services provide pannanga, "
                "website develop pannuvangala, "
                "mobile application build pannuvangala, "
                "Android support irukka, "
                "iOS support irukka, "
                "website database connect panna mudiyuma, "
                "website redesign panna mudiyuma, "
                "startup ku website develop pannuvangala."

            ),

            # ------------------------------------------------
            # Prevent previous transcription influence
            # ------------------------------------------------

            condition_on_previous_text=False,

            temperature=0.0,

            # ------------------------------------------------
            # We don't use Whisper VAD here.
            #
            # Browser handles recording.
            # ------------------------------------------------

            vad_filter=False,

            no_speech_threshold=0.70,

            log_prob_threshold=-1.5,

            compression_ratio_threshold=2.4

        )


        # ====================================================
        # DETECTED LANGUAGE
        # ====================================================

        if info is not None:

            detected_language = getattr(
                info,
                "language",
                None
            )

            language_probability = getattr(
                info,
                "language_probability",
                None
            )

            if detected_language:

                if language_probability is not None:

                    print(
                        f"🌐 Detected language: "
                        f"{detected_language} "
                        f"({language_probability:.2f})"
                    )

                else:

                    print(
                        f"🌐 Detected language: "
                        f"{detected_language}"
                    )


        # Convert generator → list

        segments = list(
            segments
        )


        # ====================================================
        # BUILD TEXT
        # ====================================================

        text_parts = []


        for segment in segments:

            if (

                segment.no_speech_prob is not None

                and

                segment.no_speech_prob
                > MAX_NO_SPEECH_PROB

            ):

                continue


            segment_text = (
                segment.text.strip()
            )


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
    # VALIDATE
    # ========================================================

    if not is_reliable_transcription(
        segments,
        text
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


    # Remove repetitions

    text = remove_repeated_words(
        text
    )


    # Correct DesFlyer

    text = correct_desflyer_words(
        text
    )


    # Correct domain terms

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
    # EXIT COMMAND
    # ========================================================

    if is_exit_command(text):

        print(
            "\n👋 Goodbye!"
        )

        return "__EXIT__"


    return text


# ============================================================
# TRANSCRIBE AUDIO BYTES
# ============================================================

def transcribe_audio_bytes(
    audio_bytes,
    suffix=".webm"
):

    if not audio_bytes:

        print(
            "❌ Empty audio received."
        )

        return ""


    temp_path = None


    try:

        # ----------------------------------------------------
        # Create temporary audio file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                audio_bytes
            )

            temp_path = (
                temp_file.name
            )


        print(
            f"💾 Browser audio received: "
            f"{len(audio_bytes)} bytes"
        )


        # ----------------------------------------------------
        # Transcribe
        # ----------------------------------------------------

        text = transcribe_audio_file(
            temp_path
        )


        return text


    except Exception as error:

        print(
            "❌ Audio processing error:",
            error
        )

        return ""


    finally:

        # ----------------------------------------------------
        # Delete temporary file
        # ----------------------------------------------------

        if (
            temp_path
            and
            os.path.exists(temp_path)
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# COMPATIBILITY FUNCTION FOR FASTAPI
# ============================================================

def speech_to_text(audio_path):

    return transcribe_audio_file(
        audio_path
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n==================================="
    )

    print(
        "🎙️ Browser Audio STT Test"
    )

    print(
        "==================================="
    )

    print(
        "\nThis STT module now expects audio"
    )

    print(
        "to be received from the browser."
    )

    print(
        "\nSupported languages:"
    )

    print(
        "🇬🇧 English"
    )

    print(
        "🇮🇳 Tamil"
    )

    print(
        "🔀 Tamil + English mixed speech"
    )

    print(
        "\nMicrophone is NOT accessed here."
    )