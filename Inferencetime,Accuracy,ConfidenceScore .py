#!/usr/bin/env python
# coding: utf-8

# # Inferencetime,Accuracy,ConfidenceScore of 100 images dataset

# ## SET-1 = 100 Diffrent faces 
# ## SET-2 = 10 faces of same person (present in SET-1)
# ## SET-3 = 10 Diffrent faces (irrespective of SET-1)

# In[54]:


'''Running SET-2 (10-same Faces) with SET-1(100-Diff Faces)'''
import cv2
import face_recognition
import os
import numpy as np
import time


overall_start = time.time()
# Load the known images and encode them
known_face_encodings = []
known_face_names = []



# Loop through the directory containing the known images
# Loop through the directory containing the known images
for file_name in os.listdir(r"D:\M.SC\HSC\100_diff_faces"):
    image = face_recognition.load_image_file(os.path.join(r"D:\M.SC\HSC\100_diff_faces", file_name))
    face_encodings = face_recognition.face_encodings(image)

    # Only use the encoding if at least one face was detected
    if len(face_encodings) > 0:
        encoding = face_encodings[0]
        known_face_encodings.append(encoding)

        # Use the file name (without extension) as the name
        name = os.path.splitext(file_name)[0]
        known_face_names.append(name)
        
        
test_images = []
for file_name in os.listdir(r"D:\M.SC\HSC\10_same_facess"):
    image = cv2.imread(os.path.join(r"D:\M.SC\HSC\10_same_facess", file_name))
    test_images.append(image)
    
inference = []
face_distances = []
num_correct = []
total_num = []
# Recognize faces in the test images
for test_image in test_images:
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    total_num.append(1)

    # Loop through each face in the test image and compare with known faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        start_main = time.time()*1000
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
                face_conf = round(np.average(face_conf),2)
                face_distances.append(face_conf)
                print(f"image matches with the {name} with confidence score of {face_conf}")
                small_end = time.time()*1000
                small_inference = small_end - small_start
                #print(f"Inference time for Image recognising is : {small_inference}MiliSec")
                
            if name == known_face_names[best_match_index]:
                num_correct.append(1)

          #If want to print the image uncomment below code
          # Draw a rectangle around the face and label the name
#         top, right, bottom, left = face_location
#         cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 1)
#         cv2.putText(test_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)



        # Display the resulting image

#     cv2.imshow('Video', test_image)
#     cv2.waitKey(0)
    end_main = time.time()*1000
    inference_time = end_main-start_main
    inference.append(round(inference_time,2))
    print(f"the Inference time for {name} is {round(inference_time,2)}miliseconds")
    print('--------------------------------')
    

small_end = time.time()*1000
small_inference = small_end - small_start
print(f"Inference time only for face recognition : {small_inference}MiliSec")
# cv2.destroyAlWindows()
overall_end = time.time()
overall_inference = str(overall_end-overall_start)
print("Overall inference time needed to execute entire program(seconds): " , overall_inference)
print("========================================")
avg_inf = sum(inference)/len(inference)
print(f"Average inference time for all the images is {avg_inf} MiliSec")
print("========================================")
accuracy = (len(num_correct)*100)/len(total_num)
print(f"Accuracy: {accuracy}%")
print("========================================")
print("List of confidence score : ",face_distances)


# 

# In[53]:


''' Running SET-3(10- diffrent faces) on SET-1(100-Diff Faces)'''
import cv2
import face_recognition
import os
import numpy as np
import time


overall_start = time.time()
# Load the known images and encode them
known_face_encodings = []
known_face_names = []

# Loop through the directory containing the known images
for file_name in os.listdir(r"D:\M.SC\HSC\100_diff_faces"):
    image = face_recognition.load_image_file(os.path.join(r"D:\M.SC\HSC\100_diff_faces", file_name))
    face_encodings = face_recognition.face_encodings(image)

    # Only use the encoding if at least one face was detected
    if len(face_encodings) > 0:
        encoding = face_encodings[0]
        known_face_encodings.append(encoding)

        # Use the file name (without extension) as the name
        known_name = os.path.splitext(file_name)[0]
        known_face_names.append(known_name)
        
        
test_images = []
for file_name in os.listdir(r"D:\M.SC\HSC\10_diff_faces"):
    image = cv2.imread(os.path.join(r"D:\M.SC\HSC\10_diff_faces", file_name))
    test_images.append(image)
    
inference = []
face_distances = []
num_correct = []
total_num = []
# Recognize faces in the test images
for test_image in test_images:
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    total_num.append(1)

    # Loop through each face in the test image and compare with known faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        start_main = time.time()*1000
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
                print(f"Inference time for Image recognising is : {small_inference}MiliSec")
                
            if name == known_face_names[best_match_index]:
                num_correct.append(1)

          #If want to print the image uncomment below code
          # Draw a rectangle around the face and label the name
#         top, right, bottom, left = face_location
#         cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 1)
#         cv2.putText(test_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)



        # Display the resulting image

#     cv2.imshow('Video', test_image)
#     cv2.waitKey(0)
    end_main = time.time()*1000
    inference_time = end_main-start_main
    inference.append(round(inference_time,2))
    print(f"the Inference time for {name} is {round(inference_time,2)}mili seconds")
    print('--------------------------------')
    


# cv2.destroyAllWindows()
overall_end = time.time()
overall_inference = str(overall_end-overall_start)
print("Overall inference time needed to execute entire program(sec): " , overall_inference)
print("========================================")
avg_inf = sum(inference)/len(inference)
print(f"Average inference time for all the images is {avg_inf} mili seconds ")
print("========================================")
accuracy = (len(num_correct)*100)/len(total_num)
print(f"Accuracy: {accuracy}%")
print("========================================")
print("List of confidence score : ",face_distances)


# In[ ]:




