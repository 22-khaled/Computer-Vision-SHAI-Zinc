#Computer Vision - SHAI - Zain Zinc YU
#1st Lec:
import cv2
cap = cv2.VideoCapture(0)
while True:
    s,img = cap.read()
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break