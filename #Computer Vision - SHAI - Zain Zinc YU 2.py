# #Computer Vision - SHAI - Zain Zinc YU
# #2nd Lec:
# import cv2
# import numpy as np

# img = cv2.imread(r"C:\Users\tp\OneDrive\Desktop\iHeal\images\LOGO.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img, (7,7), 1) #increasing the number == increases Alpha (change 1 to 4).
# ken = np.ones((5,5), np.uint8) 
# img_canny = cv2.Canny(img, 200, 200) #This numbers for Threeshold1, 2.
# cropped_image = img[10:90,50:150]
# new_image = cv2.resize(img, (500, 500))
# painted_upon_image = cv2.rectangle(img, (100, 100, 50, 50), (255, 0, 0), 5)
# painted_upon_image1 = cv2.circle(img, (100, 100), 10, (0, 0, 255))
# #painted_upon_image2 = cv2.line(painted_upon_image1, 0, 250, (0, 0, 0)) - Find the error

# cv2.imshow('image', img)
# cv2.imshow('Gray Image', img_gray)
# cv2.imshow('Blur Image', img_blur)
# cv2.imshow('Blur Image', img_canny)
# cv2.imshow('resized Image', new_image)
# cv2.imshow('crop Image', cropped_image) 
# cv2.imshow('crop Image', painted_upon_image) #(0-255) B G R (Colors).
# cv2.imshow('crop Image', painted_upon_image1) 
# #cv2.imshow('crop Image', painted_upon_image2) 

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Code from Teachable machine:
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r"C:\Users\tp\Downloads\converted_keras\keras_Model.h5", compile=False)

# Load the labels
class_names = open(r"C:\Users\tp\Downloads\converted_keras\labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
