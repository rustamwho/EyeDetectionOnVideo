import multiprocessing as mp
import cv2
from FaceDetectPytorch import FaceAndLandmarksDetectPyTorch
import time

length = int(cv2.VideoCapture("Large.MOV").get(cv2.CAP_PROP_FRAME_COUNT))
global frame_jump
global num_processes, gpu
gpu = True
num_processes = 2
"""
if gpu is True:
    frame_jump = length // 8
else:
    frame_jump = cv2.VideoCapture("Large.MOV").get(cv2.CAP_PROP_FRAME_COUNT) // num_processes  # // mp.cpu_count()
"""
cap = cv2.VideoCapture("Large.MOV")
frame_jump = cap.get(
    cv2.CAP_PROP_FRAME_COUNT) // num_processes  # по сколько кадров каждому процессу
global odd
# если кадры не делятся поровну на количество процессов
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
    """
    if (gpu is True) and (group_number == num_processes - 1):
        device = 'cuda'
        frames = frames * 7
    else:
        device = 'cpu'
    """
    device = 'cuda'
    print("group_number " + str(group_number) + "  frames  " + str(frames))
    # out = cv2.VideoWriter("output_{}.avi".format(group_number), ...)
    while proc_frames < frames:
        ret, frame = cap.read()
        print(str(startFrame + proc_frames) + " Frame")
        if frame is not None:
            frame = cv2.flip(frame, 0)
            FaceAndLandmarksDetectPyTorch(device, frame, startFrame + proc_frames)
        proc_frames += 1
    return None


if __name__ == "__main__":
    start_time = time.clock()
    mp.Pool(num_processes).map(process_video, range(num_processes))
    print("Itog --- %s seconds ---" % (time.clock() - start_time))
