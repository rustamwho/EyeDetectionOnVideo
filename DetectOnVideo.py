import numpy as np
import cv2
# from FaceDetect import FaceAndLandmarksDetect
import time
from FaceDetectPytorch import FaceAndLandmarksDetectPyTorch
from multiprocessing import Pool
import multiprocessing

FRAME_QUEUE = multiprocessing.JoinableQueue()
RESULT_QUEUE = multiprocessing.Queue()
NWORKERS = 2

cap = cv2.VideoCapture("exampleVideo.mp4")
# out = cv2.VideoWriter('output.avi', -1,20.0,(1280,720))
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 60, (3840, 2160))
start_time = time.clock()

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# for i in range(1, length):
global m
m = 0
t = 0

"""
def ReadFrames():
    frames = []
    global m
    for k in range(1, length):
        ret, frame = cap.read()
        if ret == True:
            # frame = cv2.flip(frame, 0)
            frames.append(frame)
            print(m)
            m = m + 1
        else:
            break
    return frames


def DetectAndWrite(frame):
    global m
    print(str(m) + " Frame")
    frame = FaceAndLandmarksDetectPyTorch(frame, m)
    m = m + 1"""


"""
while m < length:
    DetectAndWrite()
"""

i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)
        # frame = FaceAndLandmarksDetect(frame, i)
        print(str(i) + " Frame")
        frame = FaceAndLandmarksDetectPyTorch(frame, i)
        i = i + 1
        #out.write(frame)
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print("Itog --- %s seconds ---" % (time.clock() - start_time))
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

"""
print("length --- %s seconds ---" % (time.clock() - start_time))
start_time = time.clock()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("width --- %s seconds ---" % (time.clock() - start_time))
start_time = time.clock()
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("height --- %s seconds ---" % (time.clock() - start_time))
start_time = time.clock()
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps --- %s seconds ---" % (time.clock() - start_time))
print(length)
print(width)
print(height)
print(fps)
"""
