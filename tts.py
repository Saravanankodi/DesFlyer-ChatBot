import os
import tempfile
import pyttsx3


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(text):

    if not text or not text.strip():

        print("⚠️ Empty text received.")

        return None

    print("\n🔊 Generating TTS audio...")

    engine = None
    output_file = None

    try:

        # ====================================================
        # CREATE TEMP WAV FILE
        # ====================================================

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ).name

        print(
            "💾 TTS output:",
            output_file
        )

        # ====================================================
        # INITIALIZE PYTTSX3
        # ====================================================

        engine = pyttsx3.init()

        # ====================================================
        # VOICE SETTINGS
        # ====================================================

        # Slightly slower for a more natural
        # conversational chatbot voice.

        engine.setProperty(
            "rate",
            155
        )

        engine.setProperty(
            "volume",
            1.0
        )

        # ====================================================
        # SELECT MICROSOFT ZIRA VOICE
        # ====================================================

        voices = engine.getProperty(
            "voices"
        )

        selected_voice = None

        if voices:

            print(
                "\n🎙️ Available voices:"
            )

            for voice in voices:

                print(
                    "   -",
                    voice.name
                )

                # ------------------------------------------------
                # Select Microsoft Zira
                # ------------------------------------------------

                if "Zira" in voice.name:

                    selected_voice = voice

            # ====================================================
            # APPLY ZIRA VOICE
            # ====================================================

            if selected_voice:

                engine.setProperty(
                    "voice",
                    selected_voice.id
                )

                print(
                    "\n⭐ Selected TTS Voice:"
                )

                print(
                    "   Name:",
                    selected_voice.name
                )

                print(
                    "   Type: Microsoft Zira Desktop"
                )

            else:

                # =================================================
                # FALLBACK VOICE
                # =================================================

                print(
                    "\n⚠️ Microsoft Zira was not found."
                )

                print(
                    "⚠️ Using the first available voice."
                )

                if voices:

                    engine.setProperty(
                        "voice",
                        voices[0].id
                    )

                    print(
                        "   Fallback:",
                        voices[0].name
                    )

        # ====================================================
        # SAVE SPEECH TO WAV
        # ====================================================

        engine.save_to_file(
            text,
            output_file
        )

        engine.runAndWait()

        # ====================================================
        # STOP ENGINE
        # ====================================================

        engine.stop()

        engine = None

        # ====================================================
        # VERIFY FILE
        # ====================================================

        if not os.path.exists(
            output_file
        ):

            print(
                "❌ TTS file was not created."
            )

            return None

        file_size = os.path.getsize(
            output_file
        )

        if file_size == 0:

            print(
                "❌ TTS file is empty."
            )

            os.remove(
                output_file
            )

            return None

        print(
            f"✅ TTS audio created: "
            f"{file_size} bytes"
        )

        return output_file

    except Exception as error:

        print(
            "❌ TTS error:",
            error
        )

        if (
            output_file
            and
            os.path.exists(
                output_file
            )
        ):

            try:

                os.remove(
                    output_file
                )

            except Exception:

                pass

        return None

    finally:

        if engine is not None:

            try:

                engine.stop()

            except Exception:

                pass


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n==================================="
    )

    print(
        "🔊 DesFlyer TTS Test"
    )

    print(
        "==================================="
    )

    test_text = (
        "Hello, this is the DesFlyer voice assistant. "
        "How can I help you today?"
    )

    output = text_to_speech(
        test_text
    )

    if output:

        print(
            "\n==================================="
        )

        print(
            "✅ TTS test successful."
        )

        print(
            "==================================="
        )

        print(
            "WAV file:",
            output
        )

    else:

        print(
            "\n==================================="
        )

        print(
            "❌ TTS test failed."
        )

        print(
            "==================================="
        )