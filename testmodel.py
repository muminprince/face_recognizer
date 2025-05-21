import cv2
import numpy as np

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model
recognizer.read("Trainer.yml")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Names corresponding to IDs
id_names = ["11", "Fayez","Fayez","SHANTOSH"]

# Open the video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Recognize faces
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 50:
            name = id_names[serial]
        else:
            name = "Unknown"

        # Display the recognized name
        cv2.putText(frame, name, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture
video.release()
cv2.destroyAllWindows()

