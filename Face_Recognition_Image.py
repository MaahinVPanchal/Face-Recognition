import cv2
from random import randrange

# Load the pre-trained face cascade classifier
trained_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load images
img = cv2.imread('armyman.png')
img1 = cv2.imread('client.png')

# Convert images to grayscale
grayscaled_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
face_coordinates = trained_face.detectMultiScale(grayscaled_img)

# Draw rectangles around the detected faces
for (x, y, w, h) in face_coordinates:  # Assuming only one face is detected
    cv2.rectangle(img1, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# Display the image with the rectangles drawn
cv2.imshow("Maahin Face Detector", img1)

# Wait for a key press to exit
cv2.waitKey()

# Print completion message
print("Code Completed")
