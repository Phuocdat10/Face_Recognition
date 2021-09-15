#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import time
import queue, threading
import array
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image, ImageDraw
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json


# In[3]:


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')


# In[4]:


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# In[5]:


def loadVggFaceModel():
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        return model


# In[6]:


def ageModel():
    model = loadVggFaceModel()

    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)
    age_model.load_weights("models/age_model_weights.h5")

    return age_model


# In[7]:


def genderModel():
    model = loadVggFaceModel()
    
    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)
    
    gender_model = Model(inputs=model.input, outputs=base_model_output)
    gender_model.load_weights("models/gender_model_weights.h5")
    
    return gender_model


# In[8]:


# Load emotion model
emotion_model = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
emotion_model.load_weights('models/facial_expression_model_weights.h5')

EMOTIONS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
# emotion_model_path = "_mini_XCEPTION.102-0.66.hdf5"


# In[9]:


thick = 2
count = 0
person = 0
count_face = 0
font_scale = 0.5
x,y,w,h = 0,0,0,0
check = False
detections=''

c_time = time.time()
age_model = ageModel()
gender_model = genderModel()
# emotion_model = load_model(emotion_model_path, compile=False)
output_index = np.array([i for i in range(0,101)])


# In[10]:

# Change 0 to use camera lap
cap = cv2.VideoCapture('data/people.mp4')
while True:
    ret,frame = cap.read()
    #Detect Face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         if person < len(faces):
#             person = len(faces)
#             cv2.putText(frame, f'ID {person}', (x+int(w/2),y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]

        # Detect Eyes
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             tmp = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
        detected_face = cv2.resize(detected_face, (224,224),interpolation = cv2.INTER_AREA)
        
        pixels = image.img_to_array(detected_face)
        pixels = np.expand_dims(pixels, axis=0)
        pixels /= 255

        #predict age
        age_distributions = age_model.predict(pixels)
        apparent_age = str(int(np.floor(np.sum(age_distributions * output_index, axis = 1))[0]))
        #predict gender

        gender_distributions = gender_model.predict(pixels)
        gender_index = np.argmax(gender_distributions)

        #predict emotion
        detected_face2 = frame[int(y):int(y+h), int(x):int(x+w)]
        detected_face2 = cv2.resize(detected_face2, (48, 48),interpolation = cv2.INTER_AREA)
        detected_face2 = cv2.cvtColor(detected_face2, cv2.COLOR_BGR2GRAY)
        pixel_emo = image.img_to_array(detected_face2)
        pixel_emo = np.expand_dims(pixel_emo, axis=0)
        pixel_emo /= 255
#         predictions = emotion_model.predict(pixel_emo)
#         max_index = np.argmax(predictions[0])
#         emotion = EMOTIONS[max_index]
        emotion = emotion_model.predict(pixel_emo)
        sad = emotion[0][4]
        fear = emotion[0][2]
        angry = emotion[0][0]
        happy = emotion[0][3]
        disgust = emotion[0][1]
        neutral = emotion[0][6]
        suprised = emotion[0][5]

        if gender_index == 0: gender = "Female"
        else: gender = "Male"
        #Change Color if emotion have maxindex
        color = [(255,255,255),
               (255,255,255),
               (255,255,255),
               (255,255,255),
               (255,255,255),
               (255,255,255),
               (255,255,255)] # DA ideal
        color[np.argmax(emotion[0])] = (0,0,255)
        
        cv2.putText(frame, f'Gender: {gender}', (x+w,y+10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thick)
        cv2.putText(frame, f'Age:{apparent_age}', (x+w,y+25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thick)
        cv2.putText(frame, 'Sad: {:.2%}'.format(sad), (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color[4], thick)
        cv2.putText(frame, 'Fear: {:.2%}'.format(fear), (x,y+h+35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color[2], thick)
        cv2.putText(frame, 'Happy: {:.2%}'.format(happy), (x,y+h+50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color[3], thick)
        cv2.putText(frame, 'Angry: {:.2%}'.format(angry), (x,y+h+65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color[0], thick)
        cv2.putText(frame, 'Neutral: {:.2%}'.format(neutral), (x,y+h+80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color[6], thick)
        cv2.putText(frame, 'Suprised: {:.2%}'.format(suprised), (x,y+h+95), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color[5], thick)
    
    #Count numbers people and time look on cam
    count_face_bf = len(faces)
    if count_face != 0:
        cv2.putText(frame, 'Time: '+str(np.round(time.time()-c_time)),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(51,92,255),1)
    else: 
        c_time = time.time()
        cv2.putText(frame, 'Time: 0',(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(51,92,255),1)
    if count_face_bf < count_face:
        person += (count_face-count_face_bf)
    count_face = count_face_bf
    cv2.putText(frame, f'person {count_face}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51,92,255), 1)
    
    cv2.rectangle(frame, (5,30),(130,80),(0, 255, 255), 1)
    tmp = cv2.resize(frame,(800,600),interpolation = cv2.INTER_AREA)
    
    try:    
        cv2.imshow('face_recognition', tmp)
    except:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
                      
cap.release()
cv2.destroyAllWindows()


# In[ ]:




