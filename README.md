# Real-Time Multimodal Emotion Recognition
*Detect emotion in local recordings!* This repository was forked from [maelfabien/Multimodal-Emotion-Recognition](https://github.com/maelfabien/Multimodal-Emotion-Recognition).

## Usage
1. Clone the project locally
2. In the `04-WebApp` directory, create a `recordings` folder and an `output` folder
2. Place a `.wav` recording (that you'd like to use emotion recognition on) into the `04-WebApp/recordings` directory
3. Open the `audio_ser_from_file.py` file and change the `input_audio` variable contents to be the name of your `.wav` file
4. In a terminal, `cd` into the WebApp folder
5. Run `pip install -r requirements.txt`
6. Run `python audio_ser_from_file.py`
7. View the output `.csv` files with the recognized emotions (per second of recording) in the `04-WebApp/output` directory