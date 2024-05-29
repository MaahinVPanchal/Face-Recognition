import cv2
from random import randrange

# Load the pre-trained face cascade classifier
trained_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize face ID counter
face_id_counter = 0

while True:
    # Read frame from webcam
    successful_frame_read, frame = webcam.read()

    # Convert frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    face_coordinates = trained_face.detectMultiScale(grayscaled_img)

    # Draw rectangles around the detected faces and add labels
    for (x, y, w, h) in face_coordinates:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
        
        # Increment face ID counter
        face_id_counter += 1
        
        # Add label with face ID
        cv2.putText(frame, f'Human {face_id_counter}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with rectangles and labels drawn
    cv2.imshow("Maahin Face Detector", frame)
    
    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == 81 or key == 113:  # 'Q' or 'q' key
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()

print("Code completed")
