#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Face detection usibg OpenCV


# In[2]:


print(cv2.__version__)


# In[7]:


import cv2

# Load the classifier file

face_cascade = cv2.CascadeClassifier('C:/Users/MAYANK/Downloads/Project_Open_CV/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
# Open the video
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




