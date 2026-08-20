import pyttsx3


# Initialize TTS engine
print("Loading TTS...")

engine = pyttsx3.init()

print("TTS loaded successfully.")


def text_to_speech(text):

    if not text:
        print("❌ No text to speak.")
        return

    print("🔊 Speaking...")

    engine.say(text)
    engine.runAndWait()

    print("✅ Speech completed.")


if __name__ == "__main__":

    text = "Hello! Welcome to DesFlyer. How can I assist you today?"

    text_to_speech(text)