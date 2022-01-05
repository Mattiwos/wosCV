import cv2
import numpy as np
import threading
import os
import PySimpleGUI as sg

#global
show = 'Frame';
#

def video():#Display
    global show

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(frame,200,200) 
        print(show + " from CV")
        if show == 'Edge':
            cv2.imshow("Main frame",edges)
        elif show == 'Frame':
            cv2.imshow("Main frame",frame)
        elif show == 'Exit':
            break


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def controls():#Controls for CV 
    global show
    ##Control UI
    sg.theme('DarkAmber')   # Add a touch of color
    layout = [
        [sg.Text("Comuter Vision Control Station")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Button("Edge")],
        [sg.Button("Motion_Detection")],
        [sg.Button("Exit")]

    ]
    controlpanel = sg.Window("CV Control", layout)
    ##

    while True:     
        event, values = controlpanel.read()
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            show = 'Exit'
            break
        elif event == 'Edge':
            show = 'Edge'
        elif event == 'Motion_Detection':
            show = 'Frame'


# creating threads
if __name__ == "__main__":
  
    # print ID of current process
    print("ID of process running main program: {}".format(os.getpid()))
  
    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))
  
    # creating threads
    t1 = threading.Thread(target=video, name='t1')
    t2 = threading.Thread(target=controls, name='t2')  
  
    # starting threads
    t1.start()
    t2.start()
  
    # wait until all threads finish
    t1.join()
    t2.join()


    #END
    cap.release()
    cv2.destroyAllWindows()
    controlpanel.close()
# out.release()
