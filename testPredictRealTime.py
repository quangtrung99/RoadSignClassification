import numpy as np
from cv2 import cv2
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

#########################

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0
font = cv2.FONT_HERSHEY_SIMPLEX

#########################

# Setup Camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Import Trained Model
model = load_model("models/Model_Trained_9_11.h5")

# Class Name
classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}


# Preprocessing
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

while True: 

    ret, imgOriginal = cap.read()

    # Preprocessing Image
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32,32))
    img = preprocessing(img)
    cv2.imshow("Preprocessing Image", img)

    img = img.reshape(1,32,32,1)
    cv2.putText(imgOriginal, "Class: ", (20, 30), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability: ", (20, 70), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Predict
    predict = model.predict(img)
    classPredict = model.predict_classes(img)

    prob = np.amax(predict)
    print(classNames[classPredict[0]])
    if prob > threshold:
        cv2.putText(imgOriginal, str(classNames[classPredict[0]]), (120, 30), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(prob*100, 2)) + "%", (180, 70), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', imgOriginal)

    if(cv2.waitKey(10) & 0xFF == ord('q')):
        break
