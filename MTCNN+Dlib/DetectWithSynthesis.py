import cv2
from facenet_pytorch import MTCNN as mtcnnPytorch
import dlib

from numpy import *
import time

detectorPytorchCpu = mtcnnPytorch(margin = 100, min_face_size=30,select_largest = False, post_process = False, keep_all = False,
                                  device = 'cpu')
detectorPytorchCuda = mtcnnPytorch(margin = 100,min_face_size=30, select_largest = False, post_process = False, keep_all = False,
                                   device = 'cuda')
detectorDlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def DetectLandmarks(image, boxes):
    # faces = detectorDlib(image)
    faces = dlib.rectangles()
    # из opencv rectangle в dlib rectangle
    for r in boxes:
        dlib.rectangles.append(faces,
                               dlib.rectangle(left = int(r[0]),
                                              top = int(r[1]),
                                              right = int(r[2]),
                                              bottom = int(r[3])))
    landmarks = []
    for face in faces:
        # Получение координат контрольных точек и их построение на изображении
        landmarks.append(predictor(image, face))
    return landmarks


def FaceAndLandmarksDetectPyTorch(device, img, i):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    start_time = time.clock()
    if (device == 'cpu'):
        method = detectorPytorchCpu
    else:
        method = detectorPytorchCuda
    boxes, probs, landmarks = method.detect(image, landmarks = True)
    print("--- %s seconds ---" % (time.clock() - start_time))
    # res = show_results(image, boxes, landmarks)
    # Visualize
    if boxes is not None:
        landmarksFromDlib = DetectLandmarks(image, boxes)
        for l in landmarksFromDlib:
            for n in range(0, 68):
                x = l.part(n).x
                y = l.part(n).y
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        for b in boxes:
            cv2.rectangle(image,
                          (b[0], b[1]),
                          (b[2], b[3]),
                          (0, 155, 255), 2)
        for l in landmarks:
            for k in range(5):
                cv2.circle(image, (l[k][0], l[k][1]), 1, (0, 155, 255), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('C:/Users/rusta/Desktop/photo/outSynthesis/6.jpg', image)


img = cv2.imread('C:/Users/rusta/Desktop/photo/6.jpg')
FaceAndLandmarksDetectPyTorch('cuda', img, 1)
