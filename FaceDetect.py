from mtcnn.mtcnn import MTCNN as mt

from Landmarks import DetectLandmarks

import cv2
import tensorflow as tf
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""
outPath = "C:/Users/landwatersun1/Deskto/OuitImages"


def FaceAndLandmarksDetect(img, i):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detector = mt(None, 100)
    start_time = time.clock()
    #    with tf.device("GPU"):
    result = detector.detect_faces(image)
    print("--- %s seconds ---" % (time.clock() - start_time))

    boxes = []
    for r in result:
        boxes.append(r['box'])

    landmarks = DetectLandmarks(image, boxes)

    for l in landmarks:
        for n in range(0, 68):
            x = l.part(n).x
            y = l.part(n).y
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

    print(result)
    for r in result:
        bounding_box = r['box']
        keypoints = r['keypoints']
        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255), 2)
        cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(outPath, f'frame_{i}.jpg'.format(i = i)), image)
    return image



q = cv2.cvtColor(cv2.imread("3people.jpg"), cv2.COLOR_BGR2RGB)
im = FaceAndLandmarksDetect(q,1)
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
scale_percent = 30  # Процент от изначального размера
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

cv2.namedWindow("image")
cv2.imshow("image", img)
cv2.waitKey(0)

