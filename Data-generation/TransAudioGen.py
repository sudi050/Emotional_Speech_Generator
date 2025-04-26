import os
from gtts import gTTS
import whisper

AUDIO_INPUT_DIR = "high_confidence_labels"
OUTPUT_DIR = "Synthetic_neutral_audio"

model = whisper.load_model("base")


def transcribe_audio_to_text(audio_path):
    """Convert speech to text using Whisper."""
    result = model.transcribe(audio_path)
    return result["text"]


def generate_audio_from_text(text, output_path):
    """Convert text to neutral speech using gTTS."""
    tts = gTTS(text=text, lang="en")
    tts.save(output_path)


def process_audio_files(input_folder):
    """Process audio files: transcribe and generate neutral audio."""
    for character_folder in os.listdir(input_folder):
        character_path = os.path.join(input_folder, character_folder)
        if not os.path.isdir(character_path):
            continue

        # Define character-specific output folders
        character_transcript_dir = os.path.join(
            OUTPUT_DIR, "transcripts", character_folder
        )
        character_audio_dir = os.path.join(OUTPUT_DIR, "neutral_audio", character_folder)
        os.makedirs(character_transcript_dir, exist_ok=True)
        os.makedirs(character_audio_dir, exist_ok=True)

        # Process each audio file in the character's folder
        process_character_audio_files(
            character_path, character_transcript_dir, character_audio_dir
        )


def process_character_audio_files(
    input_folder, transcript_output_dir, neutral_audio_output_dir
):
    for file_name in os.listdir(input_folder):
        if file_name.endswith((".mp3", ".wav")):
            input_file_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]

            # Step 1: Transcribe original audio
            transcript = transcribe_audio_to_text(input_file_path)
            transcript_path = os.path.join(transcript_output_dir, f"{base_name}.txt")
            with open(transcript_path, "w") as f:
                f.write(transcript)
            print(f"Transcript saved: {transcript_path}")

            # Step 2: Generate neutral audio from transcript
            neutral_audio_path = os.path.join(
                neutral_audio_output_dir, f"{base_name}_neutral.mp3"
            )
            generate_audio_from_text(transcript, neutral_audio_path)
            print(f"Neutral audio generated: {neutral_audio_path}")


if __name__ == "__main__":
    process_audio_files(AUDIO_INPUT_DIR)
