# import cv2
# import pandas as pd
# from ultralytics import YOLO
# import cvzone
# import numpy as np
# # from tracker import *
# model=YOLO('yolov8n.pt')


# import math


# class Tracker:
#     def __init__(self):
#         # Store the center positions of the objects
#         self.center_points = {}
#         # Keep the count of the IDs
#         # each time a new object id detected, the count will increase by one
#         self.id_count = 0


#     def update(self, objects_rect):
#         # Objects boxes and ids
#         objects_bbs_ids = []

#         # Get center point of new object
#         for rect in objects_rect:
#             x, y, w, h = rect
#             cx = (x + x + w) // 2
#             cy = (y + y + h) // 2

#             # Find out if that object was detected already
#             same_object_detected = False
#             for id, pt in self.center_points.items():
#                 print("cx=",cx,"cy=",cy,"id=",id,"pt=",pt)
#                 # break
#                 dist = math.hypot(cx - pt[0], cy - pt[1])

#                 if dist < 35:
#                     self.center_points[id] = (cx, cy)
# #                    print(self.center_points)
#                     objects_bbs_ids.append([x, y, w, h, id])
#                     same_object_detected = True
#                     break
#             # break
#             # New object is detected we assign the ID to that object
#             if same_object_detected is False:
#                 self.center_points[self.id_count] = (cx, cy)
#                 objects_bbs_ids.append([x, y, w, h, self.id_count])
#                 self.id_count += 1

#         # Clean the dictionary by center points to remove IDS not used anymore
#         new_center_points = {}
#         for obj_bb_id in objects_bbs_ids:
#             _, _, _, _, object_id = obj_bb_id
#             center = self.center_points[object_id]
#             new_center_points[object_id] = center

#         # Update dictionary with IDs not used removed
#         self.center_points = new_center_points.copy()
#         return objects_bbs_ids


# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         point = [x, y]
#         # print(point)
  
        

# # cv2.namedWindow('RGB')
# # cv2.setMouseCallback('RGB', RGB)
# cap=cv2.VideoCapture('Cars.mp4')


# my_file = open("coco.txt", "r")
# data = my_file.read()
# class_list = data.split("\n") 
# #print(class_list)

# count=0

# tracker=Tracker()
# tracker1=Tracker()
# tracker2=Tracker()
# cy1=184
# cy2=209
# offset=8
# upcar={}
# downcar={}
# countercarup=[]
# countercardown=[]
# downbus={}
# counterbusdown=[]
# upbus={}
# counterbusup=[]
# downtruck={}
# countertruckdown=[]
# while True:    
#     ret,frame = cap.read()
#     if not ret:
#         break
#     count += 1
#     # if count % 2 != 0:
#     #     continue
#     frame=cv2.resize(frame,(1020,500))
#     results=model.predict(frame)
#  #   print(results)
#     a=results[0].boxes.data
#     px=pd.DataFrame(a).astype("float")
#     # if 3 not in [ int(i[-1]) for i  in a ]:
#     #     continue
    
#     # print(px)
#     # break
#     list=[]
#     list1=[]
#     list2=[]
#     for index,row in px.iterrows():
#         # print(row)
#         # break
#         x1=int(row[0])
#         y1=int(row[1])
#         x2=int(row[2])
#         y2=int(row[3])
#         d=int(row[5])
#         c=class_list[d]
#         if 'car' in c:
#            list.append([x1,y1,x2,y2])
          
#         # elif'bus' in c:
#         #     list1.append([x1,y1,x2,y2])
          
#         # elif 'truck' in c:
#         #      list2.append([x1,y1,x2,y2])
#         # if 'motorcycle' in c:
#         #     list.append([x1,y1,x2,y2])

#         # if 'person' in c:
#         #     list.append([x1,y1,x2,y2])

            
#     # break
#     bbox_idx=tracker.update(list)
#     # break
#     for bbox in bbox_idx:
#         x3,y3,x4,y4,id1=bbox
#         cx3=int(x3+x4)//2
#         cy3=int(y3+y4)//2
       
#         cv2.circle(frame,(cx3,cy3),4,(255,0,0),-1)
#         cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
#         cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
              
                  
#     print("cy1",cy1,"cy2",cy2)
#     # cv2.line(frame,(0,190),(1018,190),(0,255,0),2)
#     cv2.line(frame,(0,400),(1016,400),(0,0,255),2)
#     # cv2.line(frame,(540,0),(540,500),(0,0,0),2)
   
#     cv2.imshow("RGB", frame)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cap.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import pandas as pd
import cvzone

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video/Cars.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=False)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,conf=0.4, iou=0.8,classes=[2])
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
    cv2.imshow("RGB", im0)
    im0 = counter.start_counting(im0, tracks)
    # video_writer.write(im0)
    if cv2.waitKey(1)&0xFF==27:
        break
    frames -= 1
    print(frames, "frames left")

cap.release()
video_writer.release()
cv2.destroyAllWindows()