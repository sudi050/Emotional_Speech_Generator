import os
import glob
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import noisereduce as nr
from pydub import AudioSegment

def remove_background_voice(input_file, output_file):
    try:
        y, sr = librosa.load(input_file, sr=None)

        if len(y) < sr // 2:
            print(f" Skipping (too short for noise profiling): {input_file}")
            os.remove(temp_wav_path)
            return

        noise_sample = y[:int(0.5 * sr)]

        print(" Reducing noise...")
        reduced_audio = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

        # Save the cleaned output
        wavfile.write(output_file, sr, (reduced_audio * 32767).astype(np.int16))
        print(f" Saved cleaned audio to: {output_file}")

        # Cleanup
        os.remove(temp_wav_path)

    except Exception as e:
        print(f" Error processing {input_file}: {e}")

input_base = "Data-generation/character_audio_clips"
output_folder = "../unlabelled_bg_removed"

input_paths = glob.glob(os.path.join(input_base, "*/audio/*.wav"))

if not input_paths:
    print(f" No wav files found in '{input_base}/*/audio/'.")
else:
    os.makedirs(output_folder, exist_ok=True)

    print(f" Found {len(input_paths)} MP3 files.")

    for input_path in input_paths:
        filename = os.path.basename(input_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, base_name + "_cleaned.wav")

        print(f"\n--- Processing: {input_path} ---")
        remove_background_voice(input_path, output_path)

    print("\n All MP3 files processed. Cleaned versions saved in '../unlabelled_bg_removed'.")
