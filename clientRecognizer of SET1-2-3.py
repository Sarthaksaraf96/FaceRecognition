#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''SET1-SET2'''
import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time

class ClientRecognizer:
    def __init__(self, encoded_path,minDetectionCon = 0.5):
        self.encoded_path = encoded_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_recognition=face_recognition
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.inference = []
        self.confidence_score=[]
        self.num_correct=[]

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

    def load_data(self,encoded_path,data_dir):
        self.known_face_encodings = np.load(self.encoded_path)
        self.known_face_names = [os.path.splitext(name)[0] for name in os.listdir(data_dir)]

    def get_names(self, image_path, tolerance=0.6):
        start_time = time.time()
        test_image = cv2.imread(image_path)
        face_locations = face_recognition.face_locations(test_image)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        # Loop through each face in the test image and compare with known faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
               # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"
            confidence = None
                # Find the index of the matched face and assign the name
            if True in matches:

                matched_index = matches.index(True)
                name = self.known_face_names[matched_index]
                confidence = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                confidence = round((1 - confidence[matched_index])*100, 2) # Convert distance to confidence score
                self.confidence_score.append(confidence)
                print(f"Clients Name :{name} ; Confidence Score: {confidence}")
            else:
                print("No image in database")
            if name == self.known_face_names[matched_index]:
                self.num_correct.append(1)
        print('--------------------------------')
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference.append(inference_time)

    def get_accuracy(self):
        return (len(self.num_correct)*100)/len(self.known_face_names)
    
    def get_confidence_score(self):
        return self.confidence_score
    
    def get_average_inference_time(self):
        return sum(self.inference)/len(self.inference)
    

def main():
    start_encoding = time.time()
    data_dir = r"D:\M.SC\HSC\10_diff_faces" #directory of Train data(which we have to encode)
    
    '''#If want to encode the Data uncomment this code
    recognizer = ClientRecognizer(encoded_path)
    face_encodings = recognizer.encode_data(data_dir)
    np.save(encoded_path, face_encodings)'''
    
    encoded_path = r"D:\M.SC\HSC\Encoded_paths\encoded_100hog.npy" #enter path of encoded data of data
    recognizer = ClientRecognizer(encoded_path)
    
    recognizer.load_data(encoded_path,data_dir)
    
    test_dir = r"D:\M.SC\HSC\10_same_facess" #directory containing test images
    
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(test_dir, filename)
            names = recognizer.get_names(image_path)
            
    print("==================================")
    
    end_encoding = time.time()
    TimeForEncoding = end_encoding-start_encoding
    print(f"overall Inference is {round(TimeForEncoding,2)} Seconds")
    
    avg_inf = recognizer.get_average_inference_time()
    print(f"Average Inference Time for each image: {round(avg_inf,2)} seconds")
    lst_confidence = recognizer.get_confidence_score()
    print(f"Average Inference Time for each image: {lst_confidence}")
    accuracy = recognizer.get_accuracy()
    print(f"Accuracy : {accuracy}%")
if __name__ =="__main__":
    main()


# In[ ]:
'''
OUTPUT - SET1-SET2
Clients Name :11 ; Confidence Score: 67.83
--------------------------------
Clients Name :11 ; Confidence Score: 56.8
--------------------------------
Clients Name :11 ; Confidence Score: 66.51
--------------------------------
Clients Name :11 ; Confidence Score: 86.41
--------------------------------
Clients Name :11 ; Confidence Score: 56.7
--------------------------------
Clients Name :11 ; Confidence Score: 47.84
--------------------------------
Clients Name :11 ; Confidence Score: 63.82
--------------------------------
Clients Name :11 ; Confidence Score: 54.69
--------------------------------
--------------------------------
--------------------------------
==================================
overall Inference is 13.72 Seconds
Average Inference Time for each image: 1.37 seconds
Average Inference Time for each image: [67.83, 56.8, 66.51, 86.41, 56.7, 47.84, 63.82, 54.69]
Accuracy : 80.0%
'''





# In[4]:


'''SET1-SET3'''
import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time

class ClientRecognizer:
    def __init__(self, encoded_path,minDetectionCon = 0.5):
        self.encoded_path = encoded_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_recognition=face_recognition
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.inference = []
        self.confidence_score=[]
        self.num_correct=[]

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

    def load_data(self,encoded_path,data_dir):
        self.known_face_encodings = np.load(self.encoded_path)
        self.known_face_names = [os.path.splitext(name)[0] for name in os.listdir(data_dir)]

    def get_names(self, image_path, tolerance=0.6):
        start_time = time.time()
        test_image = cv2.imread(image_path)
        face_locations = face_recognition.face_locations(test_image)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        # Loop through each face in the test image and compare with known faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
               # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"
            confidence = None
                # Find the index of the matched face and assign the name
            if True in matches:

                matched_index = matches.index(True)
                name = self.known_face_names[matched_index]
                confidence = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                confidence = round((1 - confidence[matched_index])*100, 2) # Convert distance to confidence score
                self.confidence_score.append(confidence)
                print(f"Clients Name :{name} ; Confidence Score: {confidence}")
            else:
                print("No image in database")
            if name == self.known_face_names:
                self.num_correct.append(1)
        print('--------------------------------')
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference.append(inference_time)

    def get_accuracy(self):
        return (len(self.num_correct)*100)/len(self.known_face_names)
    
    def get_confidence_score(self):
        return self.confidence_score
    
    def get_average_inference_time(self):
        return sum(self.inference)/len(self.inference)
    

def main():
    start_encoding = time.time()
    data_dir = r"D:\M.SC\HSC\10_diff_faces" #directory of Train data(which we have to encode)
    
    '''#If want to encode the Data uncomment this code
    recognizer = ClientRecognizer(encoded_path)
    face_encodings = recognizer.encode_data(data_dir)
    np.save(encoded_path, face_encodings)'''
    
    encoded_path = r"D:\M.SC\HSC\Encoded_paths\encoded_100hog.npy" #enter path of encoded data of data
    recognizer = ClientRecognizer(encoded_path)
    
    recognizer.load_data(encoded_path,data_dir)
    
    test_dir = r"D:\M.SC\HSC\10_diff_faces" #directory containing test images
    
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(test_dir, filename)
            names = recognizer.get_names(image_path)
            
    print("==================================")
    
    end_encoding = time.time()
    TimeForEncoding = end_encoding-start_encoding
    print(f"overall Inference is {round(TimeForEncoding,2)} Seconds")
    
    avg_inf = recognizer.get_average_inference_time()
    print(f"Average Inference Time for each image: {round(avg_inf,2)} seconds")
    lst_confidence = recognizer.get_confidence_score()
    print(f"Average Inference Time for each image: {lst_confidence}")
    accuracy = recognizer.get_accuracy()
    print(f"Accuracy : {accuracy}%")
if __name__ =="__main__":
    main()


'''
OUTPUT- SET1-SET33
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
No image in database
--------------------------------
==================================
overall Inference is 24.35 Seconds
Average Inference Time for each image: 2.43 seconds
Average Inference Time for each image: []
Accuracy : 0.0%
'''





# In[ ]:




