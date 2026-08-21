import pyttsx3
import time


def text_to_speech(text):

    if not text or not text.strip():
        return

    print("🔊 Speaking...")

    engine = None

    try:

        # Create a fresh engine every time
        engine = pyttsx3.init()

        # Voice settings
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)

        # Get available voices
        voices = engine.getProperty("voices")

        if voices:
            engine.setProperty("voice", voices[0].id)

        # Add speech
        engine.say(text)

        # IMPORTANT:
        # Wait until speech is completely finished
        engine.runAndWait()

        # Give Windows audio system time to finish
        time.sleep(0.5)

        print("✅ Speech completed.")

    except Exception as error:

        print("❌ TTS error:", error)

    finally:

        # Stop and destroy this engine
        if engine is not None:

            try:
                engine.stop()
            except Exception:
                pass

            engine = None