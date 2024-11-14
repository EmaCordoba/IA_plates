import cv2
from ultralytics import YOLO
import time

model = YOLO("./runs/detect/train/weights/best.pt")

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    start_time = time.time()
    ret, frame = cam.read()

    # Realizar la detección
    results = model(frame,classes=[0])

    # Procesar cada detección para dibujar recuadros y mostrar la confianza
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas del recuadro de detección
        clase = int(result.cls[0])                 # Clase del objeto detectado

        confianza = result.conf[0].item()
        # Dibujar el recuadro (color verde y grosor de 2 píxeles)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
        # Escribir la confianza encima del recuadro
        label = f" Confianza:%{confianza*100:.2f} Clase:{model.names.get(clase)}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()