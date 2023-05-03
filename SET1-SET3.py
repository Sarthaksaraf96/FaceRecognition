#!/usr/bin/env python
# coding: utf-8

# In[7]:


'''Class for Calculating InferenceTime , ConfidenceScore and Accuracy '''
import face_recognition
import numpy as np
import cv2
import mediapipe as mp
import time
import os

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def process(self,train_path,test_path):
        start_main = time.time()*1000
        encoding_start = time.time()
        # Load the known images and encode them
        known_face_encodings = []
        known_face_names = []
        for file_name in os.listdir(train_path):
            image = face_recognition.load_image_file(os.path.join(train_path, file_name))
            face_encodings = face_recognition.face_encodings(image)

            # Only use the encoding if at least one face was detected
            if len(face_encodings) > 0:
                encoding = face_encodings[0]
                known_face_encodings.append(encoding)

                # Use the file name (without extension) as the name
                name = os.path.splitext(file_name)[0]
                known_face_names.append(name)
        encoding_end = time.time()
        encoding_time = encoding_end-encoding_start
        print(f"Inference time for encoding of images : {round(encoding_time,2)} Seconds")
        print('--------------------------------')
        test_images = []
        for file_name in os.listdir(test_path):
            image = cv2.imread(os.path.join(test_path, file_name))
            test_images.append(image)

        inference = []
        face_distances = []
        num_correct = []
        total_num = []
        for test_image in test_images:
            face_locations = face_recognition.face_locations(test_image)
            face_encodings = face_recognition.face_encodings(test_image, face_locations)
            total_num.append(1)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                small_start = time.time()*1000
                # Compare face encoding with known face encodings
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Find the index of the matched face and assign the name
                if True in matches:
                    matched_indexes = [index for index, value in enumerate(matches) if value == True]
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        face_conf = face_recognition.face_distance(known_face_encodings, face_encoding)
                        face_conf = round(np.average(face_conf),2)
                        face_distances.append(face_conf)
                        print(f"image matches with the {name} with confidence score of {face_conf}")
                        small_end = time.time()*1000
                        small_inference = small_end - small_start
                        print(f"Inference time for Recognizing face is : {small_inference}MiliSec")
                        print('--------------------------------')

                    if name == known_face_names[best_match_index]:
                        num_correct.append(1)
        end_main = time.time()*1000
        inference_time = end_main-start_main
        inference.append(round(inference_time,2))
        print(':::::::::::::::::::::::::::::::::::::')
        avg_inf = sum(inference)/len(inference)
        print(f"Average inference time for all the images is {round((int(avg_inf)/1000),2)} Seconds")
        print("========================================")
        accuracy = (len(num_correct)*100)/len(total_num)
        print(f"Accuracy: {accuracy}%")
        print("========================================")
        print("List of confidence score : ",face_distances)
        print("========================================")

def main():
    overall_start = time.time()
    detector = FaceDetector()
    
    detector.process(train_path =r"D:\M.SC\HSC\100_diff_faces" ,test_path =r"D:\M.SC\HSC\10_diff_faces")
    overall_end = time.time()
    overall_inference = overall_end-overall_start
    print(f"Overall inference time needed to execute entire program {round(overall_inference,2)} Seconds")
    print("========================================")

if __name__ =="__main__":
    main()


# In[ ]:




