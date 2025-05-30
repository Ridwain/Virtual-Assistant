import sounddevice as sd
import numpy as np
import whisper
import queue
import sys
from datetime import datetime
from scipy.signal import resample

# Load Whisper model
model = whisper.load_model("base")

# Audio parameters
SAMPLE_RATE = 48000           # Your device sample rate
TARGET_SR = 16000             # Whisper's expected sample rate
BLOCK_DURATION = 1            # seconds per audio chunk
CHANNELS = 3                  # 2 system channels + 1 mic channel
YOUR_NAME = "Ridwain"

# Volume thresholds and ratio for voice detection
MIC_VOLUME_THRESHOLD = 0.05      # minimum mic volume to consider as voice
SYSTEM_VOLUME_THRESHOLD = 0.03   # minimum system volume to consider as speech
VOLUME_RATIO = 4.0               # mic volume must be this factor above system volume

# Thread-safe queue for audio blocks
q = queue.Queue()

def list_audio_devices():
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] >= CHANNELS:
            print(f"{idx}: {dev['name']} (Input channels: {dev['max_input_channels']})")

def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio from orig_sr to target_sr"""
    number_of_samples = round(len(audio) * float(target_sr) / orig_sr)
    resampled_audio = resample(audio, number_of_samples)
    return resampled_audio

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ Audio error: {status}", file=sys.stderr)
    q.put(indata.copy())

def is_mic_voice(mic_vol, sys_vol, mic_thresh=MIC_VOLUME_THRESHOLD, ratio=VOLUME_RATIO):
    """
    Return True if mic volume is above threshold and sufficiently louder than system audio.
    """
    print(sys_vol,ratio)
    if mic_vol < mic_thresh:
        return False
    if mic_vol < sys_vol * ratio:
        return False
    return True

def is_system_voice(sys_vol, sys_thresh=SYSTEM_VOLUME_THRESHOLD):
    """
    Return True if system audio volume is above threshold.
    """
    return sys_vol > sys_thresh

def record_and_transcribe(device_id):
    print(f"\n🎙 Listening on device ID {device_id} with {CHANNELS} channels at {SAMPLE_RATE}Hz")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            device=device_id,
            callback=audio_callback,
            dtype='float32'
        ):
            print("✅ Recording started... Press Ctrl+C to stop.")

            while True:
                audio_data = []
                chunks_per_block = int(SAMPLE_RATE / 1024 * BLOCK_DURATION)
                for _ in range(chunks_per_block):
                    audio_data.append(q.get())
                audio_block = np.concatenate(audio_data, axis=0)

                # Separate system (stereo) and mic (mono) audio
                system_audio = np.mean(audio_block[:, :2], axis=1)
                mic_audio = audio_block[:, 2]

                mic_volume = np.max(np.abs(mic_audio))
                sys_volume = np.max(np.abs(system_audio))

                timestamp = datetime.now().strftime("%H:%M:%S")
                texts = []

                # Decide if mic audio is valid voice
                mic_voice = is_mic_voice(mic_volume, sys_volume)

                # Decide if system audio is voice
                sys_voice = is_system_voice(sys_volume)

                # Transcribe mic voice if detected
                if mic_voice:
                    print(f"🎤 Mic detected (vol: {mic_volume:.4f})")
                    mic_resampled = resample_audio(mic_audio, SAMPLE_RATE, TARGET_SR)
                    whisper_input = whisper.pad_or_trim(mic_resampled)
                    result = model.transcribe(whisper_input, fp16=False, language="en")
                    if result["text"].strip():
                        texts.append(f"👤 [{YOUR_NAME}]: {result['text'].strip()}")

                # Transcribe system audio if detected
                if sys_voice:
                    print(f"🖥 System audio detected (vol: {sys_volume:.4f})")
                    sys_resampled = resample_audio(system_audio, SAMPLE_RATE, TARGET_SR)
                    whisper_input = whisper.pad_or_trim(sys_resampled)
                    result = model.transcribe(whisper_input, fp16=False, language="en")
                    if result["text"].strip():
                        texts.append(f"🖥 [System]: {result['text'].strip()}")

                # Print and save results
                for line in texts:
                    print(f"📝 [{timestamp}] {line}")
                    with open("meeting_transcript.txt", "a") as f:
                        f.write(f"[{timestamp}] {line}\n")

    except KeyboardInterrupt:
        print("\n🛑 Recording stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_audio_devices()
    device_input = input("\nEnter the device ID to use for recording: ")
    try:
        device_id = int(device_input)
    except ValueError:
        print("❌ Invalid device ID. Exiting.")
        sys.exit(1)

    record_and_transcribe(device_id)
