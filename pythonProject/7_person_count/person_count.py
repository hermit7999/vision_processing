from ultralytics import YOLO
import cv2
import cvzone
import math
# from sort import*

cap  = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)
# cap = cv2.VideoCapture("D:/Python_practices/vision_processing/pythonProject/video/WalkOnRome.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")

className = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
 'scissors', 'teddy bear', 'toothbrush', 'hair drier']

# mask = cv2.imread("mask.png")

while True:
    success, img = cap.read()
    # imgRegion = cv2.bitwise_and(img,mask)

    results = model(img, stream=True) 

    # detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2,y2  = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1
            
            # cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,255),3)
            cvzone.cornerRect(img, (x1,y1,w,h))
            conf = math.ceil(box.conf[0]*100) / 100
            # print(conf)
            cls = int(box.cls[0])
            # cvzone.putTextRect(img, f'{conf}',(max(0,x1),max(35,y1-20)))
            cvzone.putTextRect(img, f'{className[cls]} {conf}',(max(0,x1),max(35,y1-20)), scale=3, thickness = 1)

            # currentClass = className[cls]
            # if currentClass == "person" and conf >= 0.3:
            #     cvzone.putTextRect(img, f'{className[cls]} {conf}',(max(0,x1),max(35,y1-20)), scale=3, thickness = 1, offset = 3)
            #     cvzone.cornerRect(img,(x1,y1,w,h),l=9)

                # currentArray = np.array([x1, y2,x2,y2,conf])
                # detections = np.vstack((detections, currentArray))
            
    # resultsTracker = tracker.update(detections)
    # for result in resultsTracker:
    #     x1,y2,x2,y2, Id = result
    #     print(result)

    cv2.imshow("Image", img)
    # cv2.imshow("Image", imgRegion)
    cv2.waitKey(1)