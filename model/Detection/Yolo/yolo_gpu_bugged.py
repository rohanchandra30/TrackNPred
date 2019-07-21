from __future__ import division

from model.Detection.Yolo.yolo_model import *
from model.Detection.Yolo.utils.utils import *
from model.Detection.Yolo.utils.datasets import *
# from yolo_model import *
# from utils.utils import *
# from utils.datasets import *

import os
import cv2
import sys
import time
import datetime
import argparse
import numpy as np
import subprocess
import re

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

IMAGE_SIZE = 416
BS = 10

def create_frames(inputDir, inputFile, framesDir):

    framesPath = os.path.join( inputDir, framesDir)
    os.makedirs(framesPath, exist_ok=True)
    inputFilePath = os.path.join(inputDir, inputFile)
    frame_savename = os.path.join(framesPath, "%06d.jpg")

    subprocess.call(["ffmpeg", "-i", inputFilePath, frame_savename])


def detect(inputDir, inputFile, framesDir, outputTxt, outputFolder, conf, nms, cuda, thread=None):

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    print("loading model...")
    model = Darknet("./resources/yolo-coco/yolov3.cfg").to(device)
    model.load_darknet_weights("./resources/yolo-coco/yolov3.weights")
    model.eval()
    print("model loaded: ", IMAGE_SIZE)

    imageOutPath = os.path.join(inputDir, outputFolder)

    classes = load_classes("./resources/yolo-coco/coco.names")  # Extracts class labels from file

    if(thread):
        thread.signalCanvas("Extracting frames from {}".format(inputFile))
    create_frames(inputDir, inputFile, framesDir)

    framesPath = os.path.join( inputDir, framesDir)

    dataloader = DataLoader(
        ImageFolder(framesPath, img_size=IMAGE_SIZE),
        batch_size=BS,
        shuffle=False,
        num_workers=0)
    
    model.eval()

    Tensor = torch.cuda.FloatTensor if cuda and torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    if thread:
        total = len(dataloader)
        thread.signalCanvas("\nPerforming object detection:")
        thread.signalTopLabel("{}: 0/{} batches".format(inputFile, total))

    prev_time = time.time()

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print(colors)

    detfile = os.path.join(inputDir, outputTxt)

    with open(detfile, "w+") as f:
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            if thread:
                tlbl = "{}: {}/{} batches".format(inputFile, batch_i + 1, total)
                thread.signalTopLabel(tlbl)
                thread.signalTopBar(max( (((batch_i + 1) / total) * 100), 1))
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, conf, nms)


            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            
            for img_i, (path, detections) in enumerate(zip(img_paths, detections)):
                print("path: ", path)
                imgFname = path.split("/")[6]
                print("imgfname: ", imgFname)
                fid = int(re.findall(r"\d+", imgFname)[0])
                
                # Create plot
                img = np.array(Image.open(path))
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(img)

                print("detections: ", detections)
                if detections is not None:
                    # Rescale boxes to original image
                    detections = rescale_boxes(detections, IMAGE_SIZE, img.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                        box_w = x2 - x1
                        box_h = y2 - y1

                        f.write("{},-1,{},{},{},{},{},-1,-1,-1,{}\n".format(fid, x1, y1, box_w, box_h, cls_conf.item(), cls_pred))

                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox)

                    # Save generated image with detections
                    plt.axis("off")
                    plt.gca().xaxis.set_major_locator(NullLocator())
                    plt.gca().yaxis.set_major_locator(NullLocator())
                    filename = path.split("/")[-1].split(".")[0]
                    plt.savefig(f"{imageOutPath}/{filename}.png", bbox_inches="tight", pad_inches=0.0)
                    print("saving image: {}".format(f"{imageOutPath}/{filename}.png"))
                    if thread:
                        thread.signalImg(f"{imageOutPath}/{filename}.png")

                    plt.close()



    # file to write to
    # f = open(os.path.join(inputDir,  outputTxt))

    # Iterate through images and save plot of detections


