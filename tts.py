import io
import os
import tempfile

from pocket_tts import TTSModel
import scipy.io.wavfile


# ============================================================
# SELECTED VOICE
# ============================================================

SELECTED_VOICE = (
    r"D:\voices_chatbot\Male\male_06_energetic.wav"
)


# ============================================================
# LOAD POCKET TTS
# ============================================================

print("\n===================================")
print("🔊 Loading Pocket TTS")
print("===================================")

tts_model = None
voice_state = None

try:

    tts_model = TTSModel.load_model()

    print("✅ Pocket TTS model loaded")

    if not os.path.exists(SELECTED_VOICE):
        raise FileNotFoundError(
            f"Selected voice not found: {SELECTED_VOICE}"
        )

    print("\n🎙️ Selected voice:")
    print("   ", SELECTED_VOICE)

    voice_state = tts_model.get_state_for_audio_prompt(
        SELECTED_VOICE
    )

    print("✅ Selected voice loaded successfully")

    print(
        f"🔊 Pocket TTS sample rate: {tts_model.sample_rate} Hz"
    )

except Exception as error:

    print(
        "❌ Pocket TTS initialization error:",
        error
    )

    tts_model = None
    voice_state = None


# ============================================================
# TEXT TO SPEECH - COMPLETE WAV
# ============================================================

def text_to_speech(text):

    if not text or not text.strip():

        print("⚠️ Empty text received.")
        return None

    print("\n🔊 Generating TTS audio...")

    output_file = None

    try:

        # ====================================================
        # CHECK TTS
        # ====================================================

        if (
            tts_model is None
            or voice_state is None
        ):

            print(
                "❌ Pocket TTS is not initialized."
            )

            return None

        # ====================================================
        # CREATE TEMP WAV FILE
        # ====================================================

        output_file = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ).name
        )

        print(
            "💾 TTS output:",
            output_file
        )

        # ====================================================
        # GENERATE COMPLETE SPEECH
        # ====================================================

        audio = tts_model.generate_audio(
            voice_state,
            text
        )

        # ====================================================
        # SAVE SPEECH TO WAV
        # ====================================================

        scipy.io.wavfile.write(
            output_file,
            tts_model.sample_rate,
            audio.numpy()
        )

        # ====================================================
        # VERIFY FILE
        # ====================================================

        if not os.path.exists(output_file):

            print("❌ TTS file was not created.")

            return None

        file_size = os.path.getsize(output_file)

        if file_size == 0:

            print("❌ TTS file is empty.")

            os.remove(output_file)

            return None

        print(
            f"✅ TTS audio created: {file_size} bytes"
        )

        return output_file

    except Exception as error:

        print("❌ TTS error:", error)

        if (
            output_file
            and os.path.exists(output_file)
        ):

            try:
                os.remove(output_file)
            except Exception:
                pass

        return None


# ============================================================
# STREAMING TTS - RAW POCKET TTS STREAM → WAV BYTES
# ============================================================

def generate_audio_stream(text):

    """
    Generate Pocket TTS audio as soon as Pocket TTS produces
    each decoded audio chunk.

    Each yielded item is a small, valid WAV byte block so the
    FastAPI WebSocket can send it immediately to the browser.
    """

    if not text or not text.strip():

        print("⚠️ Empty streaming TTS text received.")
        return

    if (
        tts_model is None
        or voice_state is None
    ):

        print("❌ Pocket TTS is not initialized.")
        return

    print("\n🔊 Pocket TTS streaming generation started...")

    try:

        audio_chunks = tts_model.generate_audio_stream(
            model_state=voice_state,
            text_to_generate=text,
            copy_state=True
        )

        chunk_number = 0

        for audio_chunk in audio_chunks:

            if audio_chunk is None:
                continue

            chunk_number += 1

            # Pocket TTS returns a 1-D tensor for streaming chunks.
            # Convert each chunk into its own valid WAV container.
            buffer = io.BytesIO()

            scipy.io.wavfile.write(
                buffer,
                tts_model.sample_rate,
                audio_chunk.numpy()
            )

            wav_bytes = buffer.getvalue()

            if not wav_bytes:
                continue

            print(
                f"🔊 Pocket TTS chunk {chunk_number}: "
                f"{len(wav_bytes)} bytes"
            )

            yield wav_bytes

        print(
            f"✅ Pocket TTS streaming completed: "
            f"{chunk_number} chunks"
        )

    except Exception as error:

        print(
            "❌ Pocket TTS streaming error:",
            error
        )

        raise


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("🔊 DesFlyer TTS Test")
    print("===================================")

    test_text = (
        "Hello, this is the DesFlyer voice assistant. "
        "How can I help you today?"
    )

    output = text_to_speech(test_text)

    if output:

        print("\n===================================")
        print("✅ TTS test successful.")
        print("===================================")
        print("WAV file:", output)

    else:

        print("\n===================================")
        print("❌ TTS test failed.")
        print("===================================")
