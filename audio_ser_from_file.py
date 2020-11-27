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

### Audio imports ###
from library.speech_emotion_recognition import *

################################################################################
########## Change this depending on where your recordings are located ##########
################################################################################
rec_dir = 'recordings/'
################################################################################

def getAudioEmotions(input_audio_filepath,output_name,SER):
    # # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(input_audio_filepath, chunk_step=step*sample_rate)

    # # Export predicted emotions to .csv format
    df = pd.DataFrame({'EMOTION': emotions, 'TIMESTAMP': timestamp})
    df.to_csv(os.path.join('output', 'audio_emotions_' + output_name + '.csv'), sep=',', index=False)

    # # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('output', 'audio_emotions_dist_' + output_name + '.csv'), sep=',')

    print('🎉 Done! 🎉')
    print('See the output files:')
    print('output/' + 'audio_emotions_' + output_name + '.csv')
    print('output/' + 'audio_emotions_dist_' + output_name + '.csv')


if __name__ == '__main__':
    # # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')
    # # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # # Loop through specific files and analyze their audio
    files_in_dir = [f for f in os.listdir(rec_dir) if os.path.isfile(os.path.join(rec_dir, f))]
    i = 0
    for f in files_in_dir:
        if f.split('.')[1] == 'm4a' or f.split('.')[1] == 'wav':
            input_audio_filepath = os.path.join(rec_dir,f)
            print(f'Reading from {input_audio_filepath}')
            output_name = f.split('.')[0]

            getAudioEmotions(input_audio_filepath, output_name, SER)
            
        i += 1
        print(f"""Number of files to go: {len(files_in_dir) - i}
            Percent files done: {i/len(files_in_dir)*100}\n""")