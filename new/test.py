from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	if not _:
		break
	results = model(frame)
	frame = results[0].plot()
	cv2.imshow("Detections", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
