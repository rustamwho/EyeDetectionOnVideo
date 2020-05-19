import cv2
from PIL import Image, ImageDraw
from Landmarks import DetectLandmarksForPyTorch

from facenet_pytorch import MTCNN as mtcnnPytorch
from matplotlib import pyplot as plt
import numpy as np
import os
from numpy import *
import time

#ima = cv2.cvtColor(cv2.imread("3people.jpg"), cv2.COLOR_BGR2RGB)
# detector = MTCNN(None, 40)
# result = detector.detect_faces(image)
detectorPytorchCpu = mtcnnPytorch(margin = 100, min_face_size=30,select_largest = False, post_process = False, keep_all = False,
                                  device = 'cpu')
detectorPytorchCuda = mtcnnPytorch(margin = 100,min_face_size=30, select_largest = False, post_process = False, keep_all = False,
                                   device = 'cuda')
outPath = "C:/Users/landwatersun1/Deskto/OuitImages"


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
        landmarksFromDlib = DetectLandmarksForPyTorch(image, boxes)
        for l in landmarksFromDlib:
            for n in range(0, 68):
                x = l.part(n).x
                y = l.part(n).y
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
        for b in boxes:
            cv2.rectangle(image,
                          (b[0], b[1]),
                          (b[2], b[3]),
                          (0, 155, 255), 2)
        for l in landmarks:
            for k in range(5):
                cv2.circle(image, (l[k][0], l[k][1]), 2, (0, 155, 255), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(outPath, f'frame_{i}.jpg'.format(i = i)), image)
    return image
    """
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline = 'white')
    for p in landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i][0] - 5.0, p[i][1] - 1.0),
                (p[i][0] + 5.0, p[i][1] + 5.0)
            ], fill = "red", outline = 'blue')
    img.show()
    """

#FaceAndLandmarksDetectPyTorch(ima, 1)
