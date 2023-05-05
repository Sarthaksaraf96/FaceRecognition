#!/usr/bin/env python
# coding: utf-8

# In[18]:


'''SET4+5 on SET1
SET4+5 = 1000 Images
SET1 = Images With 100 diffrent Faces irrespective of Face SET5'''
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
                face_locations =  self.face_recognition.face_locations(image, model="hog")
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
            if name ==self.known_face_names[matched_index]:
                self.num_correct.append(1)
        print('--------------------------------')
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference.append(inference_time)

    def get_accuracy(self):
        print(self.num_correct)
        
        return (len(self.num_correct)*100)/len(self.known_face_names)
    
    def get_confidence_score(self):
        return self.confidence_score
    
    def get_average_inference_time(self):
        return sum(self.inference)/len(self.inference)
    

def main():
    start_encoding = time.time()
    data_dir = r"D:\M.SC\HSC\1000_faces" #directory of Train data(which we have to encode)
    encoded_path = r"D:\M.SC\HSC\Encoded_paths\Encoding_hog_1000.npy"
    test_dir = r"D:\M.SC\HSC\SRK" #directory containing test images
    
    '''#If want to encode the Data uncomment this code
    recognizer = ClientRecognizer(encoded_path)
    face_encodings = recognizer.encode_data(data_dir)
    np.save(encoded_path, face_encodings)
     #enter path of encoded data of data'''
    recognizer = ClientRecognizer(encoded_path)
    
    recognizer.load_data(encoded_path,data_dir)
    
    
    
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





# In[5]:


'''SET4+5 on SET2
SET4+5 = 1000 Images
SET12 = Images With 100 diffrent Faces irrespective of Face SET5'''
import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import time

class ClientRecognizer:
    def __init__(self, encoded_path,minDetectionCon = 0.4):
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
                face_locations =  self.face_recognition.face_locations(image, model="hog")
                face_encodings =  self.face_recognition.face_encodings(image, face_locations)

                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])
                    
        
        return np.array(self.known_face_encodings)
        print (self.known_face_names)

    def load_data(self,encoded_path,data_dir):
        self.known_face_encodings = np.load(self.encoded_path)
        self.known_face_names = [os.path.splitext(name)[0] for name in os.listdir(data_dir)]

    def get_names(self, image_path, tolerance=0.4):
        start_time = time.time()
        test_image = cv2.imread(image_path)
        face_locations = face_recognition.face_locations(test_image)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        # Loop through each face in the test image and compare with known faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
               # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"
#             confidence = None
                # Find the index of the matched face and assign the name
            if True in matches:

                matched_index = matches.index(True)
                name = self.known_face_names[matched_index]
                confidence = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                confidence = round((1 - confidence[matched_index])*100, 2) # Convert distance to confidence score
                self.confidence_score.append(confidence)
                return f"Clients Name :{name} ; Confidence Score: {confidence}"
            else:
                return "No image in database"
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
    data_dir = r"D:\M.SC\HSC\1000_faces" #directory of Train data(which we have to encode)
    encoded_path = r"D:\M.SC\HSC\Encoded_paths\Encoding_hog_1000.npy"
    test_dir = r"D:\M.SC\HSC\100_diff_faces" #directory containing test images
    
    '''#If want to encode the Data uncomment this code
    recognizer = ClientRecognizer(encoded_path)
    face_encodings = recognizer.encode_data(data_dir)
    np.save(encoded_path, face_encodings)
     #enter path of encoded data of data'''
    recognizer = ClientRecognizer(encoded_path)
    
    recognizer.load_data(encoded_path,data_dir)
    
    
    null=[]
    total = []
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(test_dir, filename)
            names = recognizer.get_names(image_path)
            if names == 'No image in database':
                null.append(1)
                total.append(1)
            elif names == None:
                null.append(1)
                total.append(1)
            else:
                total.append(1)
                print(names)
            
    print("==================================")
    print(f"No image in database count :{sum(null)}")
    end_encoding = time.time()
    TimeForEncoding = end_encoding-start_encoding
    print(f"overall Inference is {round(TimeForEncoding,2)} Seconds")
    
    avg_inf = recognizer.get_average_inference_time()
    print(f"Average Inference Time for each image: {round(avg_inf,2)} seconds")
    lst_confidence = recognizer.get_confidence_score()
    print(f"list of confidence scores: {lst_confidence}")
    accuracy = (len(null)*100)/(len(total))
    print(f"Accuracy : {accuracy}%")
if __name__ =="__main__":
    main()


# In[ ]:





# In[3]:


'''SET4+5 on SET2(part_1)
SET4+5 = 1000 Images
SET 2(part_1) = Images With 10 Same Faces of diffrent persective with One Face Similiar with SET4+5'''
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
                face_locations =  self.face_recognition.face_locations(image, model="hog")
                face_encodings =  self.face_recognition.face_encodings(image, face_locations)

                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])
                    
        
        return np.array(self.known_face_encodings)
        print (self.known_face_names)

    def load_data(self,encoded_path,data_dir):
        self.known_face_encodings = np.load(self.encoded_path)
        self.known_face_names = [os.path.splitext(name)[0] for name in os.listdir(data_dir)]

    def get_names(self, image_path, tolerance=0.5):
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
    data_dir = r"D:\M.SC\HSC\1000_faces" #directory of Train data(which we have to encode)
    encoded_path = r"D:\M.SC\HSC\Encoded_paths\Encoding_hog_1000.npy"
    test_dir = r"D:\M.SC\HSC\10_same_facess" #directory containing test images
    
    '''#If want to encode the Data uncomment this code
    recognizer = ClientRecognizer(encoded_path)
    face_encodings = recognizer.encode_data(data_dir)
    np.save(encoded_path, face_encodings)
     #enter path of encoded data of data'''
    recognizer = ClientRecognizer(encoded_path)
    
    recognizer.load_data(encoded_path,data_dir)
    
    
    
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




