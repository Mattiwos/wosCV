import cv2
import numpy as np
import _thread
import time
from PIL import Image

cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter.fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
a = 0
b = 0
#multithreading
bgthreshhold = [0,40,40]
old = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a+=1
    edges = cv2.Canny(frame,200,200)
    cv2.imshow("Edges",edges)

    # if a >= 1:
    #     cimg = frame#cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    #     cheight, cwidth, cchannels = cimg.shape
    # if a == 20:
    #     background = frame#cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    #     bheight, bwidth, bchannels = background.shape
    #     shadow = frame#cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    # if a >= 21:
    #     cv2.imshow('Shadow',shadow)
    #     if a >= 22:
    #             #new Calculations
    #         bg = np.array(background)
    #
    #         ci = np.array(cimg)
    #         diff = bg - ci
    #         diff = np.absolute(diff)
    #
    #         shad = np.greater(diff,bgthreshhold)
    #         diff = np.where(diff == True, [0,0,0],[255,255,255])
    #         diff = diff.astype(np.uint8)
    #         print(diff)
    #
    #
    #         #shad = np.where(shad ==False,255, shad)
    #         #shad = np.where(shad ==True,0, shad)
    #
    #
    #         ##shad.shape = (2764800,2764800)
    #         ##shads = Image.fromarray(shad,'RGB')
    #
    #         # shad = np.place(shad,shad == 1, [0,0,0])
    #         # shad = np.place(shad,shad == 0, [255,255,255])
    #         #
    #         shadow = diff
    #                         #Previous Calculations
    #
    #             # for i in range(int(bwidth)):
    #             #     for e in range(bheight):
    #             #         if i >= 719:
    #             #             pass
    #             #         else:
    #             #
    #             #             print(np.array_equal(background[i,e],cimg[i,e]))
    #             #             print(i)
    #             #             print(e)
    #             #             r = abs(background[i,e][0] - cimg[i,e][0])
    #             #             g = abs(background[i,e][0] - cimg[i,e][1])
    #             #             b = abs(background[i,e][0] - cimg[i,e][2])
    #             #             if r >= 60 or g >=60 or b >=60:
    #             #
    #             #
    #             #                 shadow[i,e] = [0,0,0] #change shadow
    #             #
    #             #             # if np.array_equal(background[i,e],cimg[i,e]) == False:
    #             #             #   shadow[i,e] = [0,0,0]
    #             #             else:
    #             #                 shadow[i,e] = [255,255,255] #background
    #
    #
    #
    #
    #
    #
    #
    #     # out.write(frame)
    cv2.imshow('Main frame',frame)

        #cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("Process time: ")
    new = time.process_time()
    print(new-old)
    old = time.process_time()


mainframe()

cap.release()
# out.release()
cv2.destroyAllWindows()
