import cv2
from facenet_pytorch import MTCNN as mtcnnPytorch
from numpy import *
import time
detectorPytorchCpu = mtcnnPytorch(margin = 100, min_face_size=30,select_largest = False, post_process = False, keep_all = False,
                                  device = 'cpu')
detectorPytorchCuda = mtcnnPytorch(margin = 100,min_face_size=30, select_largest = False, post_process = False, keep_all = False,
                                   device = 'cuda')


def FaceAndLandmarksDetectPyTorch(device, img, i):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    start_time = time.clock()
    if (device =='cpu'):
        method = detectorPytorchCpu
    else:
        method = detectorPytorchCuda
    boxes, probs, landmarks = method.detect(image, landmarks = True)
    print("--- %s seconds ---" % (time.clock() - start_time))
    # res = show_results(image, boxes, landmarks)
    # Visualize
    if boxes is not None:
        for b in boxes:
            cv2.rectangle(image,
                          (b[0], b[1]),
                          (b[2], b[3]),
                          (0, 155, 255), 2)
        for l in landmarks:
            for k in range(5):
                cv2.circle(image, (l[k][0], l[k][1]), 2, (255, 0, 0), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/rusta/Desktop/photo/outMTCNN/7.jpg', image)

img = cv2.imread('C:/Users/rusta/Desktop/photo/7.jpg')
FaceAndLandmarksDetectPyTorch('cpu',img, 1)