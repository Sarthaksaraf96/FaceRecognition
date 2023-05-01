#!/usr/bin/env python
# coding: utf-8

# # Class for face_recognition

# In[6]:


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
        
        
    def findFaces(self , img , draw = True):
    
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []
        if self.results.detections:
            for id ,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih ,iw , ic = img.shape
                bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih),                       int(bboxC.width *iw),int(bboxC.height * ih)
                bboxs.append([id , bbox , detection.score])
                
                self.fancyDraw(img,bbox)
                cv2.putText (img , f'{int(detection.score[0] *100)}%' 
                             , (bbox[0], bbox[1] - 20) , cv2.FONT_HERSHEY_PLAIN, 
                             2 ,(255,0,255),2)

        return img , bboxs
    
    def fancyDraw(self,img,bbox ,l = 30 , t = 10):
        x,y,w,h = bbox
        x1,y1 = x+w , y+h
        
        cv2.rectangle(img , bbox , (255,0,255) , 2)
        cv2.line(img , (x,y) , (x+l , y) , (255 , 0 ,255), t )
        cv2.line(img , (x,y) , (x , y+l) , (255 , 0 ,255), t )
        
        return img

    def process(self):
        inference = []
        confidence_score=[]
        
        known_face_encodings = []
        known_face_names = []

        # Loop through the directory containing the known images
        for file_name in os.listdir(r"D:\M.SC\HSC\known"):
            image = face_recognition.load_image_file(os.path.join(r"D:\M.SC\HSC\known", file_name))
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)

            # Use the file name (without extension) as the name
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
            print(image.shape)

        # Load the test images and find faces in them
        test_images = []
        for file_name in os.listdir(r"D:\M.SC\HSC\unknown"):
            image = cv2.imread(os.path.join(r"D:\M.SC\HSC\unknown", file_name))
            test_images.append(image)


            
        for test_image in test_images:
            start_main = time.time()*1000
            face_locations = face_recognition.face_locations(test_image)
            face_encodings = face_recognition.face_encodings(test_image, face_locations)

            # Loop through each face in the test image and compare with known faces
            for face_encoding, face_location in zip(face_encodings, face_locations):

                # Compare face encoding with known face encodings
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"
                
#                 for i in test_images:
#                     end_main = time.time()
#                     print("inference Time : " + str(end_main-start_main))

                # Find the index of the matched face and assign the name
                if True in matches:

                    matched_index = matches.index(True)
                    name = known_face_names[matched_index]

                    reference_encoding = np.array(known_face_encodings)
                    target_encoding = np.array(face_encodings)
                    distance = face_recognition.face_distance([reference_encoding], target_encoding)[0]
                    confidence_score = 1 / (1 + distance)
                    #print("Distance:", distance)
                    confidence = confidence_score[0]
                    print("Confidence score:", confidence)


                # Draw a rectangle around the face and label the name
                top, right, bottom, left = face_location
                cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(test_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
            end_main = time.time()*1000
            inf = int(end_main-start_main)
            inference.append(round(inf,1))
            
            
            # Display the resulting image
            cv2.imshow("Face Recognition", test_image)
            cv2.waitKey(0)
        #print(confidence_score)
            
        cv2.destroyAllWindows()
        print("Inference time for each image(in mSec) : " , inference)
        
def main():
    Overall_start = time.time()
    detector = FaceDetector()
    
    detector.process()
    Overall_end = time.time()
    print("Overall time needed(in Sec) : ",Overall_end-Overall_start)

if __name__ =="__main__":
    main()
    


# In[2]:





# In[4]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[62]:


data ="0.89613634 0.83687138 0.87153576 0.77831797 0.84257511 0.74832722 0.84349531 0.73372446 0.86959563 0.81155868 0.88277794 0.79550436"



formatted_arr = np.array2string(data, separator=',')
print(formatted_arr)


# In[ ]:




