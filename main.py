import cv2
from pytube import YouTube
import face_recognition
from datetime import datetime

# Replace 'https://youtu.be/kusY9S8BkMU' with the actual YouTube video URL
youtube_url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA'
# Get the YouTube video object
yt = YouTube(youtube_url)
# Get the stream URL of the video
stream_url = yt.streams.filter(file_extension='mp4').first().url


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# OpenCV window
cv2.namedWindow("YouTube Video", cv2.WINDOW_NORMAL)

# Open the video stream
cap = cv2.VideoCapture(stream_url)
# Load known faces
known_face_encodings = []
known_face_names = []
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Read and display frames
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try: 
        ##faces = faceCascade.detectMultiScale(gray, 1.3, 5,minSize=(30, 30))
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        # for (x,y,w,h) in faces:
        #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #     roi_gray = gray[y:y+h, x:x+w]
        #     roi_color = img[y:y+h, x:x+w]
        # print(str(datetime.now()))
    # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match is found, use the name of the known face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                # If no match is found, add the new face to the known faces
                known_face_encodings.append(face_encoding)
                name = f"Person{len(known_face_encodings)}"  # Assign a default name
                known_face_names.append(name)
                cv2.imwrite("faces_detected/{}.jpg".format(datetime.now()),frame)

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
        cv2.imshow('Video', frame)
        #cv2.imwrite("faces_detected/{}.jpg".format(datetime.now()),img)
  
        # status = cv2.imwrite('faces_detected/faces_detected.jpg', img)
    except:
        pass
    #cv2.imshow('img',img)
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    if not ret:
        print("Error: Failed to capture frame.")
        break

    # cv2.imshow('YouTube Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
