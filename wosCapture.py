import cv2
import pafy
from mtcnn.mtcnn import MTCNN
# Load the pre-trained MTCNN model

from datetime import datetime
import numpy as np
import threading
import queue
# import argparse;

class wosCapture:
    #url youtube link or caputre camera source
    #option declares wether to use camera or outside source
    # 1, 2 ,3 make into ENUM
    def __init__(self, url, option=0):
        self.detector = MTCNN()
        self.exit_flag = False
        if option == 1:
            cv2.namedWindow("Local Video", cv2.WINDOW_NORMAL)
            self.cap = cv2.VideoCapture(url)
                    # Get the height and width of the video capture
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
                # Replace 'https://youtu.be/kusY9S8BkMU' with the actual YouTube video URL
            youtube_url = 'https://www.youtube.com/watch?v=vjBHqyt1OcQ'
            #youtube_url = 'https://www.youtube.com/watch?v=PDTmRgc2zQc'
                # Get the YouTube video object
            video = pafy.new(youtube_url)
                # Get the stream URL of the video
            self.best_stream = video.getbest(preftype="mp4")
            print(self.best_stream)
                # # Access the width and height attributes of the stream
            self.width = int(self.best_stream.dimensions[1])
            self.height = int(self.best_stream.dimensions[0])
            print(f"Video Width: {self.width}, Height: {self.height}")
                # OpenCV window
            cv2.namedWindow("YouTube Video", cv2.WINDOW_NORMAL)
                # Open the video stream
            self.cap = cv2.VideoCapture(self.best_stream.url)
            cv2.resizeWindow("YouTube Video", self.width, self.height)   

            ## Tests to optimize opencv delays 
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size to reduce delay
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print("initalized correctly");
    def start(self, treadcount=1):
        # Create a thread array
        self.threads = []
        self.NUM_THREADS = treadcount;
        MAX_QUEUE_SIZE = 10  # Adjust as needed

        self.thread_safe_q = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        try: 
            # Turn-on the worker thread.
                # Create and start threads
            for i in range(self.NUM_THREADS):
                thread = threading.Thread(target=self.thread_worker, args=(i,), daemon=True)
                thread.start()
                self.threads.append(thread)

            while True: ## create a thread pool to handle this
                ret, frame = self.cap.read()
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.thread_safe_q.put(frame)
                # print(f"queue size: {thread_safe_q.qsize()}")
                if self.thread_safe_q.full():
                    self.thread_safe_q.get()
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
            self.cap.release()
            cv2.destroyAllWindows()

            # Block until all tasks are done.
            self.thread_safe_q.join()
            # Wait for all threads to complete
            for thread in self.threads:
                thread.join()
    # threads.join()
    def thread_worker(self, thread_id):
        # global width, height
        # print(f"Video Width: {width}, Height: {height}")
        print("Thread worker {} beinning work".format(thread_id))

        while not self.exit_flag:
            frame = self.thread_safe_q.get() ## switched

            # frame = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect_faces(frame)
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
            self.thread_safe_q.task_done()

if __name__ == "__main__":
    capture = wosCapture("https://www.youtube.com/watch?v=cH7VBI4QQzA");
    capture.start()
    print("check")