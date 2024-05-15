from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import pandas as pd
import cvzone

## Load YOLO model
model = YOLO("yolov8x.pt")

## give path of video
cap = cv2.VideoCapture("video/car_truck.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("output video/cat_truck.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)


while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    ## this method track and return bounding box from frame
    tracks = model.track(im0, 
                         persist=True, 
                         show=False, 
                         conf=0.7, 
                         iou=0.5, 
                         classes=[2,3,6])
    a=tracks[0].boxes.data
    if len(a)!=0:
        px=pd.DataFrame(a).astype("float")
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[4])
            class_id=int(row[5])
            print(class_id)
            # base on classification create bounding boxes
            if class_id==2:
                cv2.rectangle(im0, 
                              (x1,y1), 
                              (x2,y2), 
                              (255,0,255), 
                              2)
                cvzone.putTextRect(im0, 
                                   f'Car:{d}', 
                                   (x1,y1), 
                                   1, 
                                   1)
            elif class_id==3:
                cv2.rectangle(im0, 
                              (x1,y1), 
                              (x2,y2), 
                              (0,0,255), 
                              2)
                cvzone.putTextRect(im0, 
                                   f'Bike:{d}', 
                                   (x1,y1), 
                                   1, 
                                   1)
            elif class_id==6:
                cv2.rectangle(im0, 
                              (x1,y1), 
                              (x2,y2), 
                              (0,255,0), 
                              2)
                cvzone.putTextRect(im0, 
                                   f'truck:{d}', 
                                   (x1,y1), 
                                   1, 
                                   1)
    video_writer.write(im0)
    if cv2.waitKey(1)&0xFF==27:
        break
    frames -= 1
    cv2.imshow("video", im0)
    print(frames, "frames left")

cap.release()
video_writer.release()
cv2.destroyAllWindows()