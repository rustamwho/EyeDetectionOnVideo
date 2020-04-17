import dlib

detectorDlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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
