import cv2
import numpy as np
import dlib
import face_recognition
from PIL import Image, ImageFilter
image = cv2.imread("image.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(image)
for face_location in face_locations:
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    face_image = Image.fromarray(face_image)
    face_image = face_image.filter(ImageFilter.BLUR)
    face_image = np.array(face_image)
    image[top:bottom, left:right] = face_image
cv2.imwrite("blurred_image.jpg", image)