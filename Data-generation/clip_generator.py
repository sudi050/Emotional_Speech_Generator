import pysrt
import re
import os

print(
    "Make sure that you have the Inside out movie SRT file and video file in the same directory as this script."
)

# Load SRT file
srt_file = "Inside.Out.2015.720p.BluRay.x264.YIFY.srt"
subs = pysrt.open(srt_file)


# Function to convert SRT time to FFmpeg-friendly format
def convert_time(timestamp, shift_ms=0):
    total_ms = (
        timestamp.hours * 3600 + timestamp.minutes * 60 + timestamp.seconds
    ) * 1000 + timestamp.milliseconds
    total_ms = max(0, total_ms + shift_ms)  # Prevent negative start times
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds = (total_ms % 60000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# Dictionary to hold clips per character with subtitle reference
character_clips = {}
character_clips["JOY"] = []
character_clips["SADNESS"] = []
character_clips["FEAR"] = []
character_clips["DISGUST"] = []
character_clips["ANGER"] = []

# Extract timestamps and cleaned text for each character — only from single-speaker subtitles
for idx, sub in enumerate(subs):
    matches = list(re.finditer(r'<font color="#808080">(.+?):</font>', sub.text))

    if len(matches) == 1:
        character = matches[0].group(1).strip().upper()
        start_time = convert_time(sub.start, shift_ms=-300)
        end_time = convert_time(sub.end, shift_ms=300)

        # Clean the subtitle text:
        raw_text = sub.text.strip()
        no_tags = re.sub(r"<.*?>", "", raw_text)  # Remove all tags like <font>, <i>, etc.

        # Remove all character name patterns like "SADNESS:", whether at start of line or mid-line
        clean_text = re.sub(r"\b[A-Z]+:\s*", "", no_tags)

        if character in character_clips:
            character_clips[character].append(
                {
                    "start": start_time,
                    "end": end_time,
                    "srt_index": idx,
                    "text": clean_text.strip(),
                }
            )

# Video file path
video_file = "Inside.Out.2015.720p.BluRay.x264.YIFY.mp4"
base_output_folder = "character_audio_clips"
os.makedirs(base_output_folder, exist_ok=True)

# print total clips for each character
for character, clips in character_clips.items():
    print(f"{character}: {len(clips)} clips")

# Create folders and extract clips
for character, clips in character_clips.items():
    char_folder = os.path.join(base_output_folder, character)
    audio_folder = os.path.join(char_folder, "audio")
    transcript_folder = os.path.join(char_folder, "transcripts")

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(transcript_folder, exist_ok=True)

    for idx, clip in enumerate(clips):
        clip_name = f"{character.lower()}_clip_{idx + 1}"
        audio_output = os.path.join(audio_folder, f"{clip_name}.wav")
        transcript_output = os.path.join(transcript_folder, f"{clip_name}.txt")

        # Skip if audio already exists
        if os.path.exists(audio_output):
            print(f"⚠️ Skipping: {clip_name} already exists.")
            continue

        # Extract audio from full video
        ffmpeg_audio_cmd = (
            f'ffmpeg -y -i "{video_file}" -ss {clip["start"]} -to {clip["end"]} '
            f'-vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_output}"'
        )
        os.system(ffmpeg_audio_cmd)

        # Write plain text transcript (no SRT formatting)

        with open(transcript_output, "w", encoding="utf-8") as txt_file:
            txt_file.write(clip["text"])

        print(f"✔ Extracted: {clip_name} for {character}")

print("✅ All audio and transcripts processed.")
