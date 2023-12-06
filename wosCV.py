import cv2
import pafy
import face_recognition
from mtcnn.mtcnn import MTCNN
# Load the pre-trained MTCNN model
detector = MTCNN()

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
    #youtube_url = 'https://www.youtube.com/watch?v=PDTmRgc2zQc'
        # Get the YouTube video object
    video = pafy.new(youtube_url)
        # Get the stream URL of the video
    best_stream = video.getbest(preftype="mp4")
    print(best_stream)
        # # Access the width and height attributes of the stream
    width = int(best_stream.dimensions[1])
    height = int(best_stream.dimensions[0])
    print(f"Video Width: {width}, Height: {height}")
        # OpenCV window
    cv2.namedWindow("YouTube Video", cv2.WINDOW_NORMAL)
        # Open the video stream
    cap = cv2.VideoCapture(best_stream.url)
    cv2.resizeWindow("YouTube Video", width, height)   

    ## Tests to optimize opencv delays 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size to reduce delay
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
exit_flag = False

def thread_worker(thread_id):
    # global width, height
    # print(f"Video Width: {width}, Height: {height}")
    print("Thread worker {} beinning work".format(thread_id))

    while not exit_flag:
        frame = thread_safe_q.get() ## switched

        print(f"Status: known_face_names: {len(known_face_names)}")
        # frame = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(frame)
        #[{'box': [1458, 596, 28, 37], 'confidence': 0.8205084800720215, 'keypoints': {'left_eye': (1471, 608), 'right_eye': (1482, 608), 'nose': (1480, 613), 'mouth_left': (1472, 622), 'mouth_right': (1481, 622)}}]
        do = 0;
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            if face['confidence'] > .9:
                do = 1
        if (do == 1):
            cv2.imwrite("faces_detected/(MM){}.jpg".format(datetime.now()),frame)
            print("Found")
        thread_safe_q.task_done()
        
# Create a thread array
threads = []
NUM_THREADS = 1;

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
        # print(f"queue size: {thread_safe_q.qsize()}")
        if thread_safe_q.full():
            thread_safe_q.get()
        # frame = thread_safe_q.get() 
        # if not ret:
        #     print("Error: Failed to capture frame.")
        #     break

        cv2.imshow('YouTube Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Ending CV")
            break
except:
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


