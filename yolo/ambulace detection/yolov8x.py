from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import pandas as pd
import cvzone


model = YOLO("YOLOv8x_a3_e100/weights/YOLOv8x_pretrained_v8x_e100_a3.pt")
cap = cv2.VideoCapture("video/Y2meta.app-Ambulance Running on Road, Free to Use this Video-(1080p50).mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Video writer
video_writer = cv2.VideoWriter("output video/Y2meta.app-Ambulance Running on Road, Free to Use this Video-(1080p50).mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
count =0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,conf=0.7, iou=0.5,classes=[2])
    a=tracks[0].boxes.data
    if len(a)!=0:
        px=pd.DataFrame(a).astype("float")
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[4])
            cv2.rectangle(im0,(x1,y1),(x2,y2),(255,0,255),2)
            cvzone.putTextRect(im0,f'{d}',(x1,y1),1,1)
    video_writer.write(im0)
    if cv2.waitKey(1)&0xFF==27:
        break
    frames -= 1
    cv2.putText(im0,str(frames) , (0, 0) , cv2.FONT_HERSHEY_SIMPLEX ,  
            1, (255, 0, 0) , 2, cv2.LINE_AA)
    cv2.imshow("video",im0)
    print(frames, "frames left")

cap.release()
video_writer.release()


# print(model.names)

# # import OS
# import os
# for x in os.listdir('test/images'):
#     image = cv2.imread('test/images/'+x)
#     results = model(image, conf=0.7, iou=0.8)
#     a=results[0].boxes.data
#     if len(a)!=0:
#         px=pd.DataFrame(a).astype("float")
#         for index,row in px.iterrows():
#             if row[5]==0:
#                 x1=int(row[0])
#                 y1=int(row[1])
#                 x2=int(row[2])
#                 y2=int(row[3])
#                 d=round(float(row[4]),2)
#                 cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),2)
#                 cvzone.putTextRect(image,f'{d}',(x1,y1),1,1)
#                 print(x)
#                 cv2.imshow("output/"+x,image)
#                 cv2.waitKey(0)
#         try:
#             cv2.imwrite("output_YOLOv8x_a3_e100/predectr_"+str(x),image)
#         except Exception as e:
#             print(px)
#             print("error:-",e)
#             print(str(x))

# cv2.destroyAllWindows()

# url="test/images/a10.webp"
# image = cv2.imread(url)
# results = model(image, conf=0.5, iou=0.8)
# a=results[0].boxes.data
# if len(a)!=0:
#     px=pd.DataFrame(a).astype("float")
#     print(px)
#     for index,row in px.iterrows():
#         if row[5]==0:
#             x1=int(row[0])
#             y1=int(row[1])
#             x2=int(row[2])
#             y2=int(row[3])
#             d=round(float(row[4]),2)
#             cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),2)
#             cvzone.putTextRect(image,f'{d}',(x1,y1),1,1)
#             cv2.imshow("output/predectr_a.jpeg",image)
#             cv2.waitKey(0)
# print("output/predectr_a.jpeg")
# cv2.imwrite("predecte_a1.jpg",image)