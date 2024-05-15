import os
import cv2
HOME = os.getcwd()
SOURCE_VIDEO_PATH = f"Cars.mp4"



# settings
MODEL = "yolov8x.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

count = 0
fontScale = 1
   
# Red color in BGR 
color = (0, 0, 255) 
  
# Line thickness of 2 px 
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX 
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    # print("xyxy:", xyxy, "confidence:", confidence, "class_id:", class_id)
    for contour,cls_id in zip(xyxy[0],class_id):
        if cls_id != 2:
            continue
        x , y, w, h = int(contour[0]), int(contour[1]), int(contour[2]), int(contour[3])
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (x, y-10), font, fontScale, color, thickness, cv2.LINE_AA)
        
    cv2.imshow('frame', frame)
    cv2.imshow('Motion Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# create frame generator
# generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# # create instance of BoxAnnotator
# box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
# # acquire first video frame
# iterator = iter(generator)
# frame = next(iterator)
# # model prediction on single frame and conversion to supervision Detections
# results = model(frame)
# detections = Detections(
#     xyxy=results[0].boxes.xyxy.cpu().numpy(),
#     confidence=results[0].boxes.conf.cpu().numpy(),
#     class_id=results[0].boxes.cls.cpu().numpy().astype(int)
# )
# # format custom labels
# labels = [
#     f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#     for _, confidence, class_id, tracker_id
#     in detections
# ]
# # annotate and display frame
# frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

