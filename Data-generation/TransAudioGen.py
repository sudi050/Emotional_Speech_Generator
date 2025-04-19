import os
from gtts import gTTS
import whisper

AUDIO_INPUT_DIR = "../data/confident_classified"
TRANSCRIPT_OUTPUT_DIR = "../data/transcripts"
NEUTRAL_AUDIO_DIR = "../data/neutral_audio"

os.makedirs(TRANSCRIPT_OUTPUT_DIR, exist_ok=True)
os.makedirs(NEUTRAL_AUDIO_DIR, exist_ok=True)

model = whisper.load_model("base")

def transcribe_audio_to_text(audio_path):
    """Convert speech to text using Whisper."""
    result = model.transcribe(audio_path)
    return result['text']

def generate_audio_from_text(text, output_path):
    """Convert text to neutral speech using gTTS."""
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)

def process_audio_files(input_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.mp3', '.wav')):
            input_file_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]

            # Step 1: Transcribe original audio
            transcript = transcribe_audio_to_text(input_file_path)
            transcript_path = os.path.join(TRANSCRIPT_OUTPUT_DIR, f"{base_name}.txt")
            with open(transcript_path, 'w') as f:
                f.write(transcript)
            print(f"Transcript saved: {transcript_path}")

            # Step 2: Generate neutral audio from transcript
            neutral_audio_path = os.path.join(NEUTRAL_AUDIO_DIR, f"{base_name}_neutral.mp3")
            generate_audio_from_text(transcript, neutral_audio_path)
            print(f"Neutral audio generated: {neutral_audio_path}")

if __name__ == "__main__":
    process_audio_files(AUDIO_INPUT_DIR)
