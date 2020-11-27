### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
from time import sleep
import re
import os
import requests
import argparse
from collections import OrderedDict

### Image processing ###
import cv2
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import dlib
from imutils import face_utils

### Model ###
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

################################################################################
########## Change this depending on where your recordings are located ##########
################################################################################
rec_dir = 'recordings/'
################################################################################

def getVideoEmotions(input_video_filepath,output_name):
    # Start video capute. 0 = Webcam, 1 = Video file, -1 = Webcam for Web
    video_capture = cv2.VideoCapture(input_video_filepath)

    # Image shape
    shape_x = 48
    shape_y = 48
    input_shape = (shape_x, shape_y, 1)

    # We have 7 emotions
    nClasses = 7

    # Timer until the end of the recording
    curr_frame_num = 0

    # Initiate Landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    # Load the pre-trained X-Ception model
    model = load_model('Models/video.h5')

    # Load the face detector
    face_detect = dlib.get_frontal_face_detector()

    # Load the facial landmarks predictor
    predictor_landmarks  = dlib.shape_predictor("Models/face_landmarks.dat")

    # Prediction vector
    predictions = []

    # Timer for length of video
    max_frame_num = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames_btwn = int(fps/3) # measure emotion every 1/3 of a sec

    # angry_0 = []
    # disgust_1 = []
    # fear_2 = []
    # happy_3 = []
    # sad_4 = []
    # surprise_5 = []
    # neutral_6 = []

    # Initialize arrays for saving predictions
    emotions = []
    face_indices = []
    timestamps = []

    # Analyze video until the end
    curr_frame_num = 0
    iter_percent = 0 # for printing
    while curr_frame_num < max_frame_num:    
        # Set the frame to be read
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_num)
        
        # Capture the frame number set above (frame here means image)
        ret, frame = video_capture.read()
        
        # Face index, face by face
        face_index = 0
        
        # Image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # All faces detected
        rects = face_detect(gray, 1)
        
        # For each detected face
        for (i, rect) in enumerate(rects):
            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y+h,x:x+w]

            # Zoom on extracted face (if face extracted)
            if face.shape[0] and face.shape[1]:
                face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))

                # Cast type float
                face = face.astype(np.float32)

                # Scale the face
                face /= float(face.max())
                face = np.reshape(face.flatten(), (1, 48, 48, 1))

                # Make Emotion prediction on the face, outputs probabilities
                prediction = model.predict(face)

                # Most likely emotion (as an index)
                prediction_result = np.argmax(prediction)

                # Convert emotion index to an emotion (string)
                emotion = ''
                if (prediction_result == 0):
                    emotion = 'Angry'
                elif (prediction_result == 1):
                    emotion = 'Disgust'
                elif (prediction_result == 2):
                    emotion = 'Fear'
                elif (prediction_result == 3):
                    emotion = 'Happy'
                elif (prediction_result == 4):
                    emotion = 'Sad'
                elif (prediction_result == 5):
                    emotion = 'Surprise'
                elif (prediction_result == 6):
                    emotion = 'Neutral'
                else:
                    emotion = 'Unknown emotion'
            else:
                emotion = 'No face extracted'
            # save results for later
            emotions.append(emotion)
            face_indices.append(face_index)
            timestamps.append(curr_frame_num/fps)

        # every so often, show percent done
        percent_done = curr_frame_num/max_frame_num*100
        if (percent_done > iter_percent):
            print('current frame: %.0f' % curr_frame_num)
            print('percent done: %.1f%%' % percent_done)
            iter_percent += 20

        # increment frame
        curr_frame_num += frames_btwn

    video_capture.release()

    # # Export predicted emotions to .csv format
    df = pd.DataFrame({'EMOTION': emotions, 'FACE_INDEX': face_indices, 'TIMESTAMP_SEC': timestamps})
    df.to_csv(os.path.join('output', 'video_emotions_' + output_name + '.csv'), sep=',', index=False)

    print('🎉 Done! 🎉')
    print('See the output file:')
    print('output/' + 'video_emotions_' + output_name + '.csv')


if __name__ == '__main__':
    # Loop through specific files and analyze their video
    files_in_dir = [f for f in os.listdir(rec_dir) if os.path.isfile(os.path.join(rec_dir, f))]
    i = 0
    for f in files_in_dir:
        if f.split('.')[1] == 'avi' or f.split('.')[1] == 'mp4':
            input_video_filepath = os.path.join(rec_dir,f)
            print(f'Reading from {input_video_filepath}')
            output_name = f.split('.')[0]

            getVideoEmotions(input_video_filepath, output_name)

        i += 1
        print(f"""Number of files to go: {len(files_in_dir) - i}
            Percent files done: {i/len(files_in_dir)*100}\n""")