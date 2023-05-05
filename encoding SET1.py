#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''overall everything
- succesfull in encoding data'''

import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class ClientRecognizer:
    def __init__(self, encoded_path,minDetectionCon = 0.5):
        self.encoded_path = encoded_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_encodings = []
        self.face_recognition=face_recognition
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.face_names = []

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

    def load_data(self):
        self.known_face_encodings = np.load(self.encoded_path)
        self.known_face_names = [os.path.splitext(name)[0] for name in os.listdir(data_dir)]

    def get_names(self, image_path, tolerance=0.6):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.results = self.faceDetection.process(image)

        if self.results.detections is not None:
                

#                 self.face_encodings = []
            for detection in self.results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih ,iw , ic = image.shape
                (left, right, top, bottom) = int(bboxC.xmin * iw) , int(bboxC.ymin * ih),                int(bboxC.width *iw),int(bboxC.height * ih)

                face_image = image[top:bottom, left:right]
                self.face_encodings.append(face_recognition.face_encodings(face_image))
            

            
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
                print(matches)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                print(face_distances)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    
                self.face_names.append(name)

            return self.face_names
        else:
            return None
        print(face_encodings)
            
def main():
    data_dir = r"D:\M.SC\HSC\100_diff_faces"
    encoded_data = r"D:\M.SC\HSC\Encoded_paths\hund.npy"
    recognizer = ClientRecognizer(encoded_data)
    face_encodings = recognizer.encode_data(data_dir)
    np.save(encoded_data, face_encodings)
    '''    
    image_path = "D:\M.SC\HSC\known\Musk.jpeg"
    recognizer = ClientRecognizer(encoded_data)
    names = recognizer.get_names(image_path)
    names'''
        

if __name__ =="__main__":
    main()

