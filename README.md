# Emotionally Expressive Speech Generation

#### This project aims to develop a system which can generate emotionally expressive speeches from text using intermediate neutral voices

#### Install the dependencies using the pyproject.toml file


---
<h2>About the dataset</h2>
Inside-Out Voice Dataset
<br><br>
Follow the procedures to run the program
Short Clips Generation

"Make sure that you have the Inside out movie SRT file and video file in the same directory as this script." <br>
expected movie_subtitle_file path - "Data-generation/Inside.Out.2015.720p.BluRay.x264.YIFY.srt" <br>
expected movie_video_file path - "Data-generation/Inside.Out.2015.720p.BluRay.x264.YIFY.mp4" <br>

Get inside Data-generation folder <br>
`cd Data-generation` <br>

Generate Clips <br>
`python3 clip_generator.py` <br><br>

Background Removal <br>
`python3 BgRemover.py` <br><br>

Labelling classifier <br>
`python3 EmotClassifier.py` <br><br>

Unlabelled-data creation <br>
`python3 generate_unlabelled_clips.py` <br><br>

Run classifier on the unlabelled_clips <br>
`python3 classify_unlabelled.py`  <br><br>


Second part of data pair - "synthetic neutrak voice" <br>
Speech - transcript - neutral  <br>
`python3 TransAudioGen.py` <br><br>


Final emotional clips generated in the folder
```Data-generation/high_confidence_labels/```
Final Synthetic Neutral clips generated in the folder
```Data-generation/Synthetic_neutral_audio/```