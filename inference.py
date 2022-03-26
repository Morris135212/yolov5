import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from IPython.core.display import display
from PIL import Image
import torchvision.transforms as T

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def infer(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',
        detect_interval=1,
        imgsz=(640, 640),  # inference size (height, width)
        vis=False,
        save_crop=False,  # save cropped prediction boxes
        line_thickness=3
        ):
    model = torch.hub.load(str(ROOT),
                           'custom',
                           path=weights,
                           source='local')
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    if source:
        v_cap = cv2.VideoCapture(source)  # initialize the video capture
    else:
        v_cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'VP90')  # define encoding type
    fps = 30.0  # define frame rate
    video_dims = imgsz  # define output dimensions
    out = cv2.VideoWriter(f"{save_dir}", fourcc, fps, video_dims)  # initialize video writer
    frame_count = 0  # initialize frame count

    while True:
        frame_count += 1  # increment frame count
        success, frame = v_cap.read()  # read frame from video
        if not success:
            raise Exception("Video Initialization error")
        if frame_count % detect_interval == 0:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert raw frame from BGR to RGB
            # img = torch.from_numpy(frame).to(device, dtype=torch.float)
            # img = img / 255
            # if len(img.shape) == 3:
            #     img = img.unsqueeze(0)  # expand for batch dim
            # pred = model(img)
            cv2.imwrite(save_dir / "tmp.png", frame)
            output = model(save_dir / "tmp.png")
            xyxy = output.pandas().xyxy[0]
            if save_crop:
                output.crop()
            for i, row in xyxy.iterrows():
                xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), line_thickness)

        out.write(frame)  # write detected frame to output video
        if vis:
            frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame_array)
            display(frame)




