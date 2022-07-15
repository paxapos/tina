import cv2

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

buscar, frame = camara.read()

if buscar == True:
  cv2.imwrite("prueba.png", frame)
  print("si")
else:
  print("no")

camara.release()