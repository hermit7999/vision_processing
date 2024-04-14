from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("D:/Python_practices/vision_processing/pythonProject/Images/3.png", show=True)
cv2.waitKey(0)
