#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Class ClientRecognizer'''

import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp

class ClientRecognizer:
    def __init__(self, encoded_path,minDetectionCon = 0.5):
        self.encoded_path = encoded_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_recognition=face_recognition
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def encode_data(self, data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(data_dir, filename)
                image =  self.face_recognition.load_image_file(image_path)
                face_locations =  self.face_recognition.face_locations(image, model="cnn")
                face_encodings =  self.face_recognition.face_encodings(image, face_locations)

                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])
                    
        
        return np.array(self.known_face_encodings)
        print (self.known_face_names)

    def load_data(self,encoded_path):
        self.known_face_encodings = np.load(self.encoded_path)
        self.known_face_names = [os.path.splitext(name)[0] for name in os.listdir(data_dir)]

    def get_names(self, image_path, tolerance=0.6):
        test_image = cv2.imread(image_path)
        face_locations = face_recognition.face_locations(test_image)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        # Loop through each face in the test image and compare with known faces
        for face_encoding, face_location in zip(face_encodings, face_locations):

            # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            # Find the index of the matched face and assign the name
            if True in matches:

                matched_index = matches.index(True)
                name = self.known_face_names[matched_index]
                print("Clients Name : ", name)
            else:
                print("No image in database")

def main():
    data_dir = "PATH\..\train" #directory of Train data(which we have to encode)
    
    '''#If want to encode the Data uncomment this code
    recognizer = ClientRecognizer(encoded_path)
    face_encodings = recognizer.encode_data(data_dir)
    np.save(encoded_path, face_encodings)'''
    
    encoded_path = "PATH\..\encoded_data.npy" #enter path of encoded data of data_dir in .npy format
    image_path = "PATH\..\client.jpeg" #test image of client
    
    recognizer = ClientRecognizer(encoded_path)
    
    recognizer.load_data(encoded_path)
    
    names = recognizer.get_names(image_path)
    names
        
if __name__ =="__main__":
    main()

