import os
import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
from scipy.signal import resample
from openai import OpenAI

# ========== OpenAI API Key ==========
os.environ["OPENAI_API_KEY"] = "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"  # <-- Replace this with your real API key

client = OpenAI()  # Now uses the key from environment variable above

# ========== TTS + Whisper ==========
tts = pyttsx3.init()
whisper_model = whisper.load_model("base")

# ========== Constants ==========
SAMPLE_RATE = 48000
TARGET_SR = 16000
QUERY_DURATION = 5  # seconds

# ========== Functions ==========
def listen_to_voice_query():
    print("\n🎤 Listening for your voice query...")
    audio = sd.rec(int(QUERY_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    audio_resampled = resample(audio, int(len(audio) * TARGET_SR / SAMPLE_RATE))
    whisper_input = whisper.pad_or_trim(audio_resampled)
    result = whisper_model.transcribe(whisper_input, fp16=False, language="en")
    query = result["text"].strip()
    print(f"🗣️ You asked: {query}")
    return query

def search_transcript(query):
    print("🔍 Searching transcript...")
    matches = []
    try:
        with open("meeting_transcript.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if any(word.lower() in line.lower() for word in query.split()):
                    matches.append(line.strip())
    except FileNotFoundError:
        print("❌ Transcript file not found.")
    return matches

def answer_with_openai(query, context):
    if not context:
        return "Sorry, I couldn't find anything useful in the transcript."
    
    context_text = "\n".join(context[-10:])
    prompt = f"""Based on the meeting transcript below, answer the user's question.

Meeting Transcript:
{context_text}

Question: {query}
Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error from OpenAI: {str(e)}"

def speak(text):
    print(f"💬 Assistant: {text}")
    tts.say(text)
    tts.runAndWait()

# ========== Main Loop ==========
def main():
    print("🔊 Virtual Meeting Assistant is running... Press Ctrl+C to exit.")
    while True:
        try:
            query = listen_to_voice_query()
            context = search_transcript(query)
            answer = answer_with_openai(query, context)
            speak(answer)
        except KeyboardInterrupt:
            print("\n🛑 Assistant exited.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            break

if __name__ == "__main__":
    main()
