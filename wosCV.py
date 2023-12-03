import cv2
import pafy
import face_recognition
import time
from datetime import datetime
import numpy as np
import threading
import queue
import argparse;

parser = argparse.ArgumentParser();
parser.add_argument('-l',help="source");
args = parser.parse_args();

if args.l is not None:
    cv2.namedWindow("Local Video", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
        # Get the height and width of the video capture
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
else:
    # Replace 'https://youtu.be/kusY9S8BkMU' with the actual YouTube video URL
    youtube_url = 'https://www.youtube.com/watch?v=cH7VBI4QQzA'
        # Get the YouTube video object
    video = pafy.new(youtube_url)
        # Get the stream URL of the video
    best_stream = video.getbest(preftype="mp4")
        # # Access the width and height attributes of the stream
    width = int(best_stream.dimensions[1])
    height = int(best_stream.dimensions[0])
    print(f"Video Width: {width}, Height: {height}")
        # OpenCV window
    cv2.namedWindow("YouTube Video", cv2.WINDOW_NORMAL)
        # Open the video stream
    cap = cv2.VideoCapture(best_stream.url)
# Load known faces
known_face_encodings = []
known_face_names = []
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


# Load the pre-trained face detection model
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
# Example using deep learning-based face detection
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
exit_flag = False

def thread_worker(thread_id):
    # global width, height
    # print(f"Video Width: {width}, Height: {height}")
    print("Thread worker {} beinning work".format(thread_id))

    while not exit_flag:
        frame = thread_safe_q.get() ## switched
        print("getting")
        # frame = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(width, height), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        try: 
            detections = net.forward()
        except:
            continue;
        # Draw rectangles around detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= .8 :  # Adjust confidence threshold as needed
                # print(confidence);

                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                # cropped_face = frame[startY:endY, startX:endX]
                # if cropped_face.size > 0:
                # cv2.imwrite("faces_detected/(FD){}.jpg".format(datetime.now()),frame)

                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                # if len(known_face_names) > 10:
                #     known_face_encodings = [];
                #     known_face_names = [];
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "unkown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    else:
                        if len(known_face_encodings) < 20:
                            known_face_encodings.append(face_encoding)
                            name = f"Person{len(known_face_encodings)}"  # Assign a default name
                            known_face_names.append(name)
                        else:
                            known_face_encodings.pop(0)
                            known_face_names.pop(0)
                            known_face_encodings.append(face_encoding)
                            name = f"Person{len(known_face_encodings)}"  # Assign a default name
                            known_face_names.append(name)

                # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cropped_face = frame[startY:endY, startX:endX]
                # if len(face_recognition.face_locations(frame)) > 0:
                    # print(confidence);
                        cv2.imwrite("faces_detected/{}.jpg".format(datetime.now()),cropped_face)
                # else:
                #     cv2.imwrite("faces_detected/(FD){}.jpg".format(datetime.now()),frame)

        thread_safe_q.task_done()
        
# Create a thread array
threads = []
NUM_THREADS = 4;

try: 
    # Turn-on the worker thread.
    # Read and display frames
    MAX_QUEUE_SIZE = 10  # Adjust as needed

    thread_safe_q = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        # Create and start threads
    for i in range(NUM_THREADS):
        thread = threading.Thread(target=thread_worker, args=(i,), daemon=True)
        thread.start()
        threads.append(thread)

    while True: ## create a thread pool to handle this
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thread_safe_q.put(frame)
        # frame = thread_safe_q.get() 
        # if not ret:
        #     print("Error: Failed to capture frame.")
        #     break

        cv2.imshow('YouTube Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Ending CV")
            break
except KeyboardInterrupt:
    print("Keyboard interrupt. Exiting gracefully.")
    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # Block until all tasks are done.
    thread_safe_q.join()
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    exit_flag = True
    # threads.join()
    print("cleaning up threads")
print('All work completed')


