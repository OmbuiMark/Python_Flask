import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import playsound
import time

def sound_alarm(path):
    # play an alarm sound file
    playsound.playsound(path)

def calculate_EAR(eye):
    # compute the Euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # compute the Euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])
    
    # calculate the eye aspect ratio (EAR)
    EAR = (A + B) / (2.0 * C)
    return EAR

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# define the threshold below which to play the alarm
EAR_THRESHOLD = 0.25

# initialize counters to keep track of frames and drowsiness
COUNTER = 0
ALARM_ON = False

# set the path to the alarm sound file
ALARM_PATH = "alarm.wav"

# start capturing video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the video feed
    ret, frame = cap.read()
    
    # resize the frame to reduce processing time
    frame = cv2.resize(frame, (640, 480))
    
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame using dlib
    faces = detector(gray, 0)
    
    # loop over the detected faces
    for face in faces:
        # detect facial landmarks using dlib
        landmarks = predictor(gray, face)
        
        # extract the left and right eye landmarks
        leftEye = []
        rightEye = []
        for i in range(36,42):
            leftEye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(42,48):
            rightEye.append((landmarks.part(i).x, landmarks.part(i).y))
        
        # calculate the eye aspect ratio (EAR) for each eye
        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)
        
        # calculate the average EAR of both eyes
        avgEAR = (leftEAR + rightEAR) / 2.0
        
        # if the average EAR falls below the threshold, increment the drowsiness counter
        if avgEAR < EAR_THRESHOLD:
            COUNTER += 1
            
            # if the alarm is not already on, start playing the alarm
            if not ALARM_ON:
                ALARM_ON = True
                t = Thread(target=sound_alarm, args=(ALARM_PATH,))
                t.deamon = True
                t.start()
                
        # otherwise, reset the counter and turn off the alarm
        else:
            COUNTER = 0
            ALARM_ON = False
        
        # draw the eyes on the frame to aid in debugging
        cv2.polylines(frame, [np.array(leftEye, np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(rightEye, np.int32)], True, (0, 255, 0), 2)
        
    # display the frame
    cv2.imshow("Frame", frame)
    
    # quit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
# release the resources used by the camera and close all windows
cap.release()
cv2.destroyAllWindows()
