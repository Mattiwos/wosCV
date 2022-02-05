import cv2
import numpy as np
import threading
import os
import PySimpleGUI as sg

#global variables
modes = ['Motion_Detection', 'Edge', 'Normal', 'Gray_Scale']
videomode = 'Normal'; #[Motion_Detection, Edge, Normal,Gray_Scale]
exitprotocol = False;
#

def video():#Display
    global videomode,exitprotocol

    cap = cv2.VideoCapture(1)
    display = 'Computer Vision Display'
    cv2.namedWindow(display, cv2.WINDOW_NORMAL)

    while (exitprotocol != True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        
        print(videomode + " from CV")

        if videomode == 'Normal':
            cv2.imshow(display,frame)
        elif videomode == 'Edge':
            edges = cv2.Canny(frame,200,200) 
            cv2.imshow(display,edges)
        elif videomode == 'Gray_Scale':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(display,gray)
        else:
            print("Unrecognized Video Mode")
            exitprotocol = True
            break



        if cv2.waitKey(1) & 0xFF == ord('q'):
            exitprotocol = True
            break
    cap.release()
    cv2.destroyAllWindows()
    

def controls():#Controls for CV 
    global videomode, exitprotocol, modes
    ##Control UI
    sg.theme('DarkAmber')   # Add a touch of color
    layout = [
        [sg.Text("Video Mode:", s=(60,1), font='Helvitica')],
        [sg.Button("Normal")],
        [sg.Button("Edge")],
        [sg.Button("Motion Detection")],
        [sg.Text("Custom Mode:")],
        [sg.Multiline(size=(30,1), key='input'),sg.Button("Submit")],
        [sg.Button("Exit")]

    ]
    controlpanel = sg.Window("Comuter Vision Control Station", layout)
    ##

    while (exitprotocol != True):     
        event, values = controlpanel.read()
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            exitprotocol = True;
            break
        elif event == 'Edge':
            videomode = 'Edge'
        elif event == 'Motion Detection':
            videomode = 'Motion_Detection'
        elif event == 'Normal':
            videomode = 'Normal'
        elif event == 'Submit' and videomode in modes:
            videomode = values['input']
            
    controlpanel.close()

# creating threads
if __name__ == "__main__":
  
    # print ID of current process
    print("ID of process running main program: {}".format(os.getpid()))
  
    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))
  
    # creating threads
    video = threading.Thread(target=video, name='Video')
    controls = threading.Thread(target=controls, name='Controls')  
  
    # starting threads
    video.start()
    controls.start()
  
    # wait until all threads finish
    video.join()
    controls.join()

