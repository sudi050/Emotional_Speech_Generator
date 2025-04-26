import os
import pysrt
import re
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import glob
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import noisereduce as nr


def clean_subtitle_text(text):
    """Remove HTML tags and speaker labels from subtitle text"""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove speaker labels like "JOY:"
    text = re.sub(r"[A-Z]+:", "", text)
    # Remove parenthetical descriptions like "(LAUGHING)"
    text = re.sub(r"\([^)]*\)", "", text)
    return text.strip()


def extract_audio_from_video(video_path, output_audio_path):
    """Extract full audio from video file"""
    print(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    video.close()
    return output_audio_path


def extract_audio_clips(audio_path, subtitles_path, output_dir, padding_ms=200):
    """Extract audio clips for each subtitle with padding"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the full audio
    full_audio = AudioFileClip(audio_path)

    # Load subtitles
    subs = pysrt.open(subtitles_path)

    print(f"Found {len(subs)} subtitles, extracting audio clips...")

    # Process each subtitle
    for i, sub in enumerate(subs):
        # Clean the subtitle text
        text = clean_subtitle_text(sub.text)
        if not text:
            continue

        # Calculate time with padding (convert to seconds)
        start_time = sub.start.ordinal / 1000 - (padding_ms / 1000)
        end_time = sub.end.ordinal / 1000 + (padding_ms / 1000)

        # Ensure start time is not negative
        start_time = max(0, start_time)
        # Ensure end time doesn't exceed audio duration
        end_time = min(full_audio.duration, end_time)

        # Extract the audio segment
        audio_segment = full_audio.subclip(start_time, end_time)

        # Create a filename based on the subtitle index and text
        # Limit the length of the text in the filename
        short_text = text[:30].replace(" ", "_").replace("/", "_").replace("\\", "_")
        short_text = re.sub(r"[^\w\s]", "", short_text)  # Remove special characters
        filename = f"{i + 1:03d}_{short_text}.wav"
        output_path = os.path.join(output_dir, filename)

        # Write the audio clip
        audio_segment.write_audiofile(output_path, logger=None)
        print(f"Created clip {i + 1}/{len(subs)}: {output_path}")

    # Close the audio file
    full_audio.close()
    print(f"All audio clips have been extracted to {output_dir}")


def main():
    video_path = "Inside.Out.2015.720p.BluRay.x264.YIFY.mp4"
    subtitles_path = "Inside.Out.2015.720p.BluRay.x264.YIFY.srt"
    output_dir = "unlabelled_clips"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    padding_ms = 200
    # Extract full audio from video
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    if not os.path.exists(audio_path):
        extract_audio_from_video(video_path, audio_path)

    # Extract audio clips based on subtitles
    extract_audio_clips(audio_path, subtitles_path, output_dir, padding_ms)


def remove_background_voice(input_file, output_file):
    try:
        y, sr = librosa.load(input_file, sr=None)

        noise_sample = y[: int(0.5 * sr)]

        print(" Reducing noise...")
        reduced_audio = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

        # Save the cleaned output
        wavfile.write(output_file, sr, (reduced_audio * 32767).astype(np.int16))
        print(f" Saved cleaned audio to: {output_file}")

    except Exception as e:
        print(f" Error processing {input_file}: {e}")


def remove_bg_main():
    input_folder = "./unlabelled_clips"
    output_folder = "./unlabelled_bg_removed"
    os.makedirs(output_folder, exist_ok=True)
    input_paths = glob.glob(os.path.join(input_folder, "*.wav"))
    print(f" Found {len(input_paths)} wav files.")

    for input_path in input_paths:
        filename = os.path.basename(input_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, base_name + "_cleaned.wav")

        print(f"\n--- Processing: {input_path} ---")
        remove_background_voice(input_path, output_path)

    print(
        "\n All .wav files processed. Cleaned versions saved in './unlabelled_bg_removed'."
    )


if __name__ == "__main__":
    main()
    remove_bg_main()
