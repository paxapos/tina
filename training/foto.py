import cv2

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    raise IOError("Cannot open webcam")

buscar, frame = camara.read()

if buscar == True:
  cv2.imwrite("prueba.png", frame)
  print("si")
else:
  print("no")

camara.release()