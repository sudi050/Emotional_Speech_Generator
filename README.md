# Emotionally Expressive Speech Generation

#### This project aims to develop a system which can generate emotionally expressive speeches from text using intermediate neutral voices
---
<h2>About the dataset</h2>
Inside-Out Voice Dataset
<br><br>
Follow the procedures to run the program
Short Clips Generation

"Make sure that you have the Inside out movie SRT file and video file in the same directory as this script." <br>
expected movie_subtitle_file path - "Data-generation/Inside.Out.2015.720p.BluRay.x264.YIFY.srt" <br>
expected movie_video_file path - "Data-generation/Inside.Out.2015.720p.BluRay.x264.YIFY.mp4" <br>

`python3 Data-generation/clip_generator.py`

Background Removal

`python3 Data-generation/BgRemover.py`

Labelling classifier

`python3 Data-generation/EmotClassifier.py`

Speech - transcript - neutral 

`python3 Data-generation/TransAudioGen.py`
