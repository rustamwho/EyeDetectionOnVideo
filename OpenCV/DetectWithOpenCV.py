import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Загружаем каскады для глаз.
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('C:/Users/rusta/Desktop/photo/7.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray,               #
        scaleFactor=1.2,    # Находим лица на фото
        minNeighbors=2,     #
        minSize=(20, 20)    #
    )

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w] # Вырезаем область с лицами
    roi_color = img[y:y + h, x:x + w]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    eyes = eyeCascade.detectMultiScale(
        roi_gray,              #
        scaleFactor=1.2,       # Ищем глаза в области с лицом
        minNeighbors=4,
        minSize=(3, 3),
    )
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Рисуем область глаз
"""
im=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
scale_percent = 30  # Процент от изначального размера
print (im.shape[1])
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)"""

cv2.imwrite('C:/Users/rusta/Desktop/photo/outOpenCV/7.jpg', img)
