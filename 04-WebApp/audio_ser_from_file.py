#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt

# ### Flask imports
# import requests
# from flask import Flask, render_template, session, request, redirect, flash, Response

### Audio imports ###
from library.speech_emotion_recognition import *

# ### Video imports ###
# from library.video_emotion_recognition import *

# ### Text imports ###
# from library.text_emotion_recognition import *
# from library.text_preprocessor import *
# from nltk import *
# from tika import parser
# from werkzeug.utils import secure_filename
# import tempfile


################################################################################
############ Change these depending on what you name the recordings ############
input_audio = 'super_happy.wav'
input_audio_filepath = os.path.join('recordings',input_audio)
output_name = input_audio.split('.')[0]
################################################################################




# # Sub dir to speech emotion recognition model
model_sub_dir = os.path.join('Models', 'audio.hdf5')

# # Instanciate new SpeechEmotionRecognition object
SER = speechEmotionRecognition(model_sub_dir)

# # Predict emotion in voice at each time step
step = 1 # in sec
sample_rate = 16000 # in kHz
emotions, timestamp = SER.predict_emotion_from_file(input_audio_filepath, chunk_step=step*sample_rate)

# # Export predicted emotions to .csv format
df = pd.DataFrame({'EMOTION': emotions, 'TIMESTAMP': timestamp})
df.to_csv(os.path.join('output', output_name + '_audio_emotions.csv'), sep=',', index=False)
# SER.prediction_to_csv(emotions, os.path.join("output", output_name + "_audio_emotions.csv"), mode='w')

# # Get most common emotion during the interview
major_emotion = max(set(emotions), key=emotions.count)

# # Calculate emotion distribution
emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

# # Export emotion distribution to .csv format for D3JS
df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
df.to_csv(os.path.join('output', output_name + '_audio_emotions_dist.csv'), sep=',')

print('🎉 Done! 🎉')
print('See the output files:')
print('output/' + output_name + '_audio_emotions.csv')
print('output/' + output_name + '_audio_emotions_dist.csv')