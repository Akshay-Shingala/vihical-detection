import os
HOME = os.getcwd()
print(HOME)
SOURCE_VIDEO_PATH = f"{HOME}/Cars_On_Highway.mp4"

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

from IPython import display
display.clear_output()


import supervision
print("supervision.__version__:", supervision.__version__)