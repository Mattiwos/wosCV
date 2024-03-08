import cv2
import pafy
from mtcnn.mtcnn import MTCNN
from datetime import datetime
import numpy as np
import threading
import queue
import argparse

class VideoProcessor:
    def __init__(self, source=None):
        self.source = source
        self.detector = MTCNN()
        self.exit_flag = False
        self.known_face_encodings = []
        self.known_face_names = []
        self.MAX_QUEUE_SIZE = 10
        self.NUM_THREADS = 1
        self.threads = []

    def start_processing(self):
        self.setup_video_capture()
        self.setup_threads()
        self.process_video()

    def setup_video_capture(self):
        if self.source is not None:
            cv2.namedWindow("Local Video", cv2.WINDOW_NORMAL)
            self.cap = cv2.VideoCapture(0)
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            youtube_url = 'https://www.youtube.com/watch?v=672RU1BRjzk'
            video = pafy.new(youtube_url)
            best_stream = video.getbest(preftype="mp4")
            self.width = int(best_stream.dimensions[1])
            self.height = int(best_stream.dimensions[0])
            cv2.namedWindow("YouTube Video", cv2.WINDOW_NORMAL)
            self.cap = cv2.VideoCapture(best_stream.url)
            cv2.resizeWindow("YouTube Video", self.width, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def setup_threads(self):
        self.thread_safe_q = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        for i in range(self.NUM_THREADS):
            thread = threading.Thread(target=self.thread_worker, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def process_video(self):
        try:
            while True:
                ret, frame = self.cap.read()
                self.thread_safe_q.put(frame)
                if self.thread_safe_q.full():
                    self.thread_safe_q.get()
                # cv2.imshow('YouTube Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Ending CV")
                    break
        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting gracefully.")
        finally:
            self.cleanup()

    def thread_worker(self, thread_id):
        print("Thread worker {} beginning work".format(thread_id))
        while not self.exit_flag:
            frame = self.thread_safe_q.get()
            faces = self.detector.detect_faces(frame)
            for face in faces:
                x, y, width, height = face['box']
                # Define the margins to expand around the detected face
                margin = 50  # Adjust this value according to your preference
                # Calculate new cropping coordinates
                x_new = max(0, x - margin)
                y_new = max(0, y - margin)
                width_new = min(frame.shape[1], width + 2 * margin)
                height_new = min(frame.shape[0], height + 2 * margin)
                # Extract the face region with wider surroundings
                face_image = frame[y_new:y_new + height_new, x_new:x_new + width_new]
                if face['confidence'] > .9:
                    try:
                        cv2.imwrite("faces_detected/(MM){}.jpg".format(datetime.now()), face_image)
                        print("Found")
                    except Exception as e:
                        print("Error:", e)
            self.thread_safe_q.task_done()


    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.thread_safe_q.join()
        self.exit_flag = True
        for thread in self.threads:
            thread.join()
        print('All work completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', help="source")
    args = parser.parse_args()

    video_processor = VideoProcessor(args.l)
    video_processor.start_processing()
