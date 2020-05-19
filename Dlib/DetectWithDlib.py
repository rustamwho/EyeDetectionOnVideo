import dlib
import cv2

detectorDlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def Detect(image):
    faces = detectorDlib(image,1)
    # Обход списка всех лиц попавших на изображение
    for face in faces:
        # Получение координат вершин прямоугольника и его построение на изображении
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Получение координат контрольных точек и их построение на изображении
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.imwrite('C:/Users/rusta/Desktop/photo/outDlib/7.jpg', image)


img = cv2.imread('C:/Users/rusta/Desktop/photo/7.jpg')
Detect(img)


def DetectFaces(image):
    faces = detectorDlib(image, 1)
    # Обход списка всех лиц попавших на изображение
    for face in faces:
        # Получение координат вершин прямоугольника и его построение на изображении
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imwrite('C:/Users/rusta/Desktop/photo/outDlib/5.jpg', image)


def DetectLandmarks(image, boxes):
    # faces = detectorDlib(image)
    faces = dlib.rectangles()
    # из opencv rectangle в dlib rectangle
    for r in boxes:
        left = int(r[0])
        top = int(r[1])
        right = int(r[0] + r[2])
        bottom = int(r[1] + r[3])
        dlib.rectangles.append(faces, dlib.rectangle(left = left, top = top, right = right, bottom = bottom))
    landmarks = []
    for face in faces:
        # Получение координат контрольных точек и их построение на изображении
        landmarks.append(predictor(image, face))
    return landmarks


def DetectLandmarksForPyTorch(image, boxes):
    # faces = detectorDlib(image)
    faces = dlib.rectangles()
    # из opencv rectangle в dlib rectangle
    for r in boxes:
        dlib.rectangles.append(faces,
                               dlib.rectangle(left = int(r[0]), top = int(r[1]), right = int(r[2]), bottom = int(r[3])))
    landmarks = []
    for face in faces:
        # Получение координат контрольных точек и их построение на изображении
        landmarks.append(predictor(image, face))
    return landmarks
