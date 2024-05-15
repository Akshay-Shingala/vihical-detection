# vihical-detection
 this repo is about detecting vehicals like car truck using YOLO model. we can train yolo with custom datasets with custom dataset it also work wheel.
 we use YOLOv8 for this repo 

 ### YOLO model 
 * yolo model is very popular for this days. becuase its offer to retrain YOLO model with custom data. also YOLO is more faster than other models like RCNN, fast RCNN, faster RCNN, etc.you can see more about it in its website [YOLO](https://docs.ultralytics.com/)

 ### You can also watch youtube video that avalable on github by codebasic 
 [![https://i.ytimg.com/vi/ag3DLKsl2vk/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLBI6DrUZyuAfAevvFCg9AMoZori2A](https://www.youtube.com/watch?v=ag3DLKsl2vk&t=469s&ab_channel=codebasics)](https://www.youtube.com/watch?v=ag3DLKsl2vk&t=469s&ab_channel=codebasics)

- hear you can also watch our output videos.

## custom train
- you can can see custome train YOLO model in ambulace detection folder.
- this model is train with our custom ambulace detection dataset.


## steps and explane how to run repo
* step 1 - Clone repo 
``` 
git clone https://github.com/Akshay-Shingala/vihical-detection.git 
```

* step 2 - create virtual envirement and activate it

```
python3 -m venv <envirement name>
```
or
```
vituealenv <envirement name>
```
Activate envirement

```
# mac and ubantu
source <envirement name>/bin/activate

# windows
<envirement name>\Script\activate
```
* step 4 - install all requirements
```
pip install -r requirements.txt
```

* step 5 - run python file
```
python3 yolov8x.py
```
