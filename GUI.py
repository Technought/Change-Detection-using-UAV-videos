import os
import numpy as np
import copy
import time
import math
import cv2
from skimage.measure import compare_ssim
from scipy.spatial import distance as dist
from collections import OrderedDict
from tkinter import *
from tkinter import filedialog

#Global Variables
yolo_weights = "./Yolo/yolov3.weights"
yolo_config = "./Yolo/yolov3.cfg"
yolo_classes = "./Yolo/coco.names"
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416
computation_backend = cv2.dnn.DNN_BACKEND_OPENCV
target_device = cv2.dnn.DNN_TARGET_OPENCL
video_rotation = ""
ref_data = []
video_data = []
ref_dir = ""
video_dir = ""
max_disappearing_ref = 0
max_disappearing_video = 0
obj_pairs = {}
score_history = {}

def yolo():
    global net, outputlayers

    # load weights and configuration files
    net = cv2.dnn.readNet(yolo_weights, yolo_config)

    # set computation backend
    net.setPreferableBackend(computation_backend)

    # set target compurting device
    net.setPreferableTarget(target_device)

    # define output layers
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    # load object classes
    with open(yolo_classes,"r") as f:
        classes = [line.strip() for line in f.readlines()]


def main():
    yolo()
    root = Tk()
    root.title("Change Detection")

    # geometry("window width x window height + position right + position down")
    root.geometry('600x600+0+0')
    # root.resizable(False, False)

    convert_Label = Label(root, text="Convert Video to Frames")
    saveto_Label = Label(root, text="Save to")
    ref_dir_label = Label(root, text="Reference video")
    video_dir_label = Label(root, text="New video")

    convert_path_entry = Entry(root, width=50, borderwidth=3)
    saveto_path_entry = Entry(root, width=50, borderwidth=3)
    ref_dir_path_entry = Entry(root, width=50, borderwidth=3)
    video_dir_path_entry = Entry(root, width=50, borderwidth=3)

    # convert_run_button = Button(root, text="Convert", command=)

    convert_Label.grid(row=0, column=0)
    convert_path_entry.grid(row=0, column=1)
    saveto_Label.grid(row=1, column=0)
    saveto_path_entry.grid(row=1, column=1)
    ref_dir_label.grid(row=2, column=0)
    ref_dir_path_entry.grid(row=2, column=1)
    video_dir_label.grid(row=3, column=0)
    video_dir_path_entry.grid(row=3, column=1)

    root.mainloop()

if __name__ == '__main__':
    main()
