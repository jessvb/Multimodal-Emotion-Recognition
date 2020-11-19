# Real-Time Multimodal Emotion Recognition
*Detect emotion in local recordings!* This repository was forked from [maelfabien/Multimodal-Emotion-Recognition](https://github.com/maelfabien/Multimodal-Emotion-Recognition).

## Usage
1. Clone the project locally
2. In the project root, create a `recordings` folder and an `output` folder
3. Place a `.wav` (for audio) or `.avi` (for video) recording (that you'd like to use emotion recognition on) into the `recordings` directory
4. Open `audio_ser_from_file.py` (for audio) or `video_er_from_file.py` (for video) and change the `input_audio` or `input_video` variable contents to be the name of your `.wav` or `.avi` file
5. In a terminal, `cd` into the project root
6. Run `pip install -r requirements.txt` (if you run into an error with `pyaudio`, you may need to do a `conda install portaudio` and then re-run the previous command)
7. Run `python audio_ser_from_file.py` for speech emotion recognition or `python video_er_from_file.py` for facial emotion recognition
8. View the output `.csv` files with the recognized emotions (per second of recording) in the `output` directory