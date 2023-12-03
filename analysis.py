import os
import cv2
import face_recognition

# Load known faces
known_face_encodings = []
known_face_names = []

mypath = "faces_detected";

onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
for file in onlyfiles:
    print(os.path.join(mypath,file))
    frame = cv2.imread(os.path.join(mypath,file))

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
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
            known_face_encodings.append(face_encoding)
            name = f"Person{len(known_face_encodings)}"  # Assign a default name
            known_face_names.append(name)
                # Draw a rectangle around the face and display the name
print(known_face_names)
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()



# import cv2
# import pafy
# import face_recognition
# from datetime import datetime
# import numpy as np
# import threading
# import queue

# # Replace 'https://youtu.be/kusY9S8BkMU' with the actual YouTube video URL
# youtube_url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA'
# # Get the YouTube video object
# video = pafy.new(youtube_url)
# # Get the stream URL of the video
# best_stream = video.getbest(preftype="mp4")
# # # Access the width and height attributes of the stream
# width = int(best_stream.dimensions[1])
# height = int(best_stream.dimensions[0])
# print(f"Video Width: {width}, Height: {height}")

# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# # OpenCV window
# cv2.namedWindow("YouTube Video", cv2.WINDOW_NORMAL)

# # Open the video stream
# cap = cv2.VideoCapture(best_stream.url)
# # Load known faces
# known_face_encodings = []
# known_face_names = []
# # Check if the video opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()

# # Read and display frames
# thread_safe_q = queue.Queue()
# # Load the pre-trained face detection model
# prototxt_path = 'deploy.prototxt'
# model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
# # Example using deep learning-based face detection
# net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# # blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
# exit_flag = False

# def thread_worker():
#     # global width, height
#     while not exit_flag:
#         frame = thread_safe_q.get() ## switched
#         # frame = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)

#         blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(width, height), mean=(104.0, 177.0, 123.0))
#         net.setInput(blob)
#         try: 
#             detections = net.forward()
#         except:
#             continue;
#         # Draw rectangles around detected faces
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.9:  # Adjust confidence threshold as needed
#                 print(confidence);

#                 box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 cv2.imwrite("faces_detected/(FD){}.jpg".format(datetime.now()),frame[startY:endY, startX:endX])

#                 face_locations = face_recognition.face_locations(frame)
#                 face_encodings = face_recognition.face_encodings(frame, face_locations)
#                 for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#                     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                     name = "unkown"
#                     if True in matches:
#                         first_match_index = matches.index(True)
#                         name = known_face_names[first_match_index]

#                     else:
#                         known_face_encodings.append(face_encoding)
#                         name = f"Person{len(known_face_encodings)}"  # Assign a default name
#                         known_face_names.append(name)
#                         print("found")
#                 # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#                         cv2.imwrite("faces_detected/(FD){}.jpg".format(datetime.now()),frame[startY:endY, startX:endX])

#         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # try: 
#         #     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
#         #     for (x,y,w,h) in faces:
#         #         # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#         #         # roi_gray = gray[y:y+h, x:x+w]
#         #         # roi_color = frame[y:y+h, x:x+w]  
#         #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#         #     # face_locations = face_recognition.face_locations(frame)
#         #     # face_encodings = face_recognition.face_encodings(frame, face_locations)
#         #     if len(faces) != 0:
#         #         cv2.imwrite("faces_detected/{}.jpg".format(datetime.now()),frame)
#         # #     # print(str(datetime.now()))
#         # # # Loop through each face found in the frame
#         #     # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         #     #     # Check if the face matches any known faces
#         #     #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         #     #     name = "Unknown"
#         #     #     # If a match is found, use the name of the known face
#         #     #     if True in matches:
#         #     #         first_match_index = matches.index(True)
#         #     #         name = known_face_names[first_match_index]
#         #     #     else:
#         #     #         print("found")
#         #     #         # If no match is found, add the new face to the known faces
#         #     #         known_face_encodings.append(face_encoding)
#         #     #         name = f"Person{len(known_face_encodings)}"  # Assign a default name
#         #     #         known_face_names.append(name)
#         #     #         # Draw a rectangle around the face and display the name
#         #     #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         #     #         font = cv2.FONT_HERSHEY_DUPLEX
#         #     #         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
#         #     #         cv2.imwrite("faces_detected/{}.jpg".format(datetime.now()),frame)


#         # # # Display the resulting frame
#         # #     cv2.imshow('Video', frame)
#         # #     #cv2.imwrite("faces_detected/{}.jpg".format(datetime.now()),img)
    
#         # #     # status = cv2.imwrite('faces_detected/faces_detected.jpg', img)
#         # except:
#         #     pass
#         thread_safe_q.task_done()
        

# try: 
#     # Turn-on the worker thread.

#     threads = threading.Thread(target=thread_worker, daemon=True)
#     threads.start()
#     while True: ## create a thread pool to handle this
#         ret, frame = cap.read()
#         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         thread_safe_q.put(frame)
#         # if not ret:
#         #     print("Error: Failed to capture frame.")
#         #     break

#         cv2.imshow('YouTube Video', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Ending CV")
#             break
# except KeyboardInterrupt:
#     print("Keyboard interrupt. Exiting gracefully.")
#     # Release the video capture object and close the OpenCV window
#     cap.release()
#     cv2.destroyAllWindows()

#     # Block until all tasks are done.
#     thread_safe_q.join()
#     thread_safe_q.close()
#     exit_flag = True
#     # threads.join()
#     print("cleaning up threads")
# print('All work completed')

