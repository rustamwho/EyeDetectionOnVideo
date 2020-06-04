import multiprocessing as mp
import cv2
from FaceDetectPytorch import FaceAndLandmarksDetectPyTorch
import time
import sys
global frame_jump
global num_processes, gpu
gpu = True
num_processes = 1
cap = cv2.VideoCapture("4K.MOV")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_jump = cap.get(
    cv2.CAP_PROP_FRAME_COUNT) // num_processes
global odd
if length % num_processes != 0:
    odd = True
else:
    odd = False


def process_video(group_number):
    global frame_jump, length, num_processes, odd, gpu, cap
    startFrame = frame_jump * group_number
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    proc_frames = 0
    if (group_number == num_processes - 1) and (odd is True) and (num_processes != 1):
        frames = length - frame_jump * (num_processes - 1)  # последнему процессу оставшиеся кадры
    else:
        frames = frame_jump
    if gpu is True:
        device = 'cuda'
    else:
        device = 'cpu'
    print("group_number " + str(group_number) + "  frames  " + str(frames))
    # out = cv2.VideoWriter("output_{}.avi".format(group_number), ...)
    while proc_frames < frames:
        start_time1 = time.clock()
        ret, frame = cap.read()
        print("--- %s seconds for reading---" % (time.clock() - start_time1))
        print(str(startFrame + proc_frames) + " Frame")
        if frame is not None:
            frame = cv2.flip(frame, 0)
            print (sys.getsizeof(frame))
            FaceAndLandmarksDetectPyTorch(device, frame, startFrame + proc_frames)
        proc_frames += 1
    return None


if __name__ == "__main__":
    start_time = time.clock()
    mp.Pool(num_processes).map(process_video, range(num_processes))
    print("Itog --- %s seconds ---" % (time.clock() - start_time))

