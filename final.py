#!/usr/bin/python

# Install Dependencies

# !pip install opencv-python==3.4.2.16
# !pip install scikit-image==0.16.2
# !pip install matplotlib
# !pip install numpy
# !pip install scipy==1.3.x

# Import Packages

import os
import numpy as np
import copy
import math
import cv2
import time
from skimage.measure import compare_ssim
from scipy.spatial import distance as dist
from collections import OrderedDict
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.messagebox as tkMessageBox
from multiprocessing import Process, Queue, Manager
from queue import Empty

# Global Variables
# manager = None

# ********************************************** Change Detection **********************************************


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def load_frames(ns, path, rot):
    cap = cv2.VideoCapture(path)
    data = []
    i = 1
    while cap.isOpened():
        if ns.abort_flag: return None
        ret, frame = cap.read()
        if ret==0:
            break
        if not type(rot) == type(None):
            frame = cv2.rotate(frame, rot)
        if(frame.shape[0] > frame.shape[1]):
            frame = image_resize(frame, height = 1280)
        else:
            frame = image_resize(frame, width = 1280)
        ns.output_image = copy.deepcopy(frame)
        data.append(frame)
        i += 1
    return data


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    class_ids = []
    final_boxes = []
    final_classes = []
    final_confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                class_ids.append(class_id)

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        final_classes.append(class_ids[i])
        final_confidences.append(confidences[i])
    return zip(final_boxes, final_confidences, final_classes)


def yolo_get_objects(ns, data):
    global net, outputlayers, classes
    objs = []
    for image in data:
        if ns.abort_flag: return None
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (ns.IMG_WIDTH, ns.IMG_HEIGHT),[0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(outputlayers)

        # Remove the bounding boxes with low confidence
        objs.append(list(post_process(image, outs, ns.CONF_THRESHOLD, ns.NMS_THRESHOLD)))
        
        ns.output_image = copy.deepcopy(image)
        for det in objs[-1]:
            for box, _, _ in det:
                left, top, width, height = tuple(box)
                cv2.rectangle(ns.output_image, (left, top), (left + width,top + height),(0, 255, 0),2)

    return objs


def compare_objs(img1, box1, img2, box2):
    width = img1.shape[0]
    height = img1.shape[1]

    left1, top1, width1, height1 = box1
    mask = np.zeros((width, height),np.uint8)
    mask = cv2.rectangle(mask, (left1, top1), (left1+width1, top1+height1), 1, thickness=-1)
    img1_masked = cv2.bitwise_and(img1, img1, mask=mask)

    left2, top2, width2, height2 = box2
    mask = np.zeros((width, height),np.uint8)
    mask = cv2.rectangle(mask, (left2, top2), (left2+width2, top2+height2), 1, thickness=-1)
    img2_masked = cv2.bitwise_and(img2, img2, mask=mask)

    orb_detector = cv2.ORB_create(5000)
    kp1, d1 = orb_detector.detectAndCompute(img2_masked, None)
    kp2, d2 = orb_detector.detectAndCompute(img1_masked, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches.sort(key=lambda x: x.distance)
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for k in range(len(matches)):
        p1[k, :] = kp1[matches[k].queryIdx].pt
        p2[k, :] = kp2[matches[k].trainIdx].pt
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    if(type(homography) == type(None)):
        return (None, None)
    transformed_img = cv2.warpPerspective(img2_masked, homography, (height, width))
    (score, diff) = compare_ssim(cv2.cvtColor(img1_masked, cv2.COLOR_BGR2GRAY),cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY), full=True)
    return (score, diff)


def couple_objectIDs(ns, data1, data2, lifetime1, lifetime2, track1, track2, mid_frames1, mid_frames2):
    ns.obj_pairs = {}
    ns.score_history = {}
    for i in range(0, len(lifetime1)):
        ns.score_history[i] = OrderedDict()
        offset = 0
        # print("t1",track1[mid_frames1[i]])
        mid1 = mid_frames1[i]
        # print(mid1)
        while i not in list(track1[mid1].keys()):
            offset += 1
            if mid1+offset < len(track1) and i in list(track1[mid1+offset].keys()):
                mid1 = mid1+offset
            if mid1-offset > 0 and i in list(track1[mid1-offset].keys()):
                mid1 = mid1-offset
        # print(mid1)
        obj1 = track1[mid1][lifetime1[i][0]][1]
        obj1_img = data1[mid1]
        for j in range(0, len(lifetime2)):
            if ns.abort_flag: return (None, None)
            offset = 0
            # print("t2",track2[mid_frames2[j]])
            mid2 = mid_frames2[j]
            # print(mid2)
            while j not in list(track2[mid2].keys()):
                offset += 1
                if mid2+offset < len(track2) and j in list(track2[mid2+offset].keys()):
                    mid2 = mid2+offset
                if mid2-offset > 0 and j in list(track2[mid2-offset].keys()):
                    mid2 = mid2-offset
            # print(mid2)
            obj2 = track2[mid2][lifetime2[j][0]][1]
            obj2_img = data2[mid2]

            (score, _) = compare_objs(obj1_img, obj1, obj2_img, obj2)
            if score:
                if score >= 0.7:
                    ns.score_history[i][j] = score

    def get_key(dict, val):
        for key, value in dict.items():
            if val == value:
                return key

    def assign_objs(objid, ignore_list):
        ignore_list.sort()
        matched_objects = list(ns.score_history[objid].keys())
        matched_objects.sort()
        if ignore_list == matched_objects:
            return
        ignored_score_history = copy.deepcopy(ns.score_history)
        for i in ignore_list:
            del ignored_score_history[objid][i]
        obj_with_max_score = max(ignored_score_history[objid], key=ignored_score_history[objid].get)
        if(obj_with_max_score not in ns.obj_pairs.values()):
            ns.obj_pairs[objid] = obj_with_max_score
        else:
            initial_assignment = get_key(ns.obj_pairs, obj_with_max_score)
            if ns.score_history[initial_assignment][obj_with_max_score] > ns.score_history[objid][obj_with_max_score]:
                ignore_list.append(obj_with_max_score)
                assign_objs(objid, ignore_list)
            else:
                ns.obj_pairs[objid] = obj_with_max_score
                ignore_list = [obj_with_max_score]
                del ns.obj_pairs[initial_assignment]
                assign_objs(initial_assignment, ignore_list)

    for i in ns.score_history.keys():
        if ns.abort_flag: return
        assign_objs(i, [])
    return (ns.obj_pairs, ns.score_history)


# centroid tracker with modifications from https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
class CentroidTracker():
    def __init__(self, maxDisappeared=0):

        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively

        # A counter used to assign unique IDs to each object
        self.nextObjectID = 0

        # A dictionary that utilizes the object ID as the key and the centroid (x, y)-coordinates as the value
        self.objects = OrderedDict()

        # Maintains number of consecutive frames (value) a particular object ID (key) has been marked as “lost”for
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking

        # The number of consecutive frames an object is allowed to be marked as “lost/disappeared” until we deregister the object.
        self.maxDisappeared = maxDisappeared

        # stores (frameNo, (centroid, box)) when a new object is detcted
        self.appearing = OrderedDict()

        # stores start and end of an object as (objectID, appearing[objectID], (frameNo-maxDisappeared,objects[objectID]))
        self.lifetime = []

        self.frameNo = 0

    def register(self, centroid, box):
        # when registering an object we use the next available object
        # ID to store the centroid and add it to appearing
        self.objects[self.nextObjectID] = (centroid, box)
        self.disappeared[self.nextObjectID] = 0
        self.appearing[self.nextObjectID]=(self.frameNo, (centroid, box))
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries and add the information to lifetime
        self.lifetime.append((objectID, self.appearing[objectID],
                             (self.frameNo-self.maxDisappeared, self.objects[objectID])))
        del self.objects[objectID]
        del self.disappeared[objectID]

    def deregister_all(self):
        objs = list(self.objects.keys())
        for objectID in objs:
            self.deregister(objectID)

    def update(self, rects):
        self.frameNo += 1
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputBoxes = {}

        # loop over the bounding box rectangles
        for (i, box) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            (startX, startY, endX, endY) = box
            cX = int((2*startX + endX) / 2.0)
            cY = int((2*startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputBoxes[tuple(inputCentroids[i])] = box

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputBoxes[tuple(inputCentroids[i])])
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = [l for l,b in list(self.objects.values())]

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = (inputCentroids[col], inputBoxes[tuple(inputCentroids[col])])
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputBoxes[tuple(inputCentroids[i])])

        # return the set of trackable objects
        return self.objects


def detect_changes(ns,l):
    ns.text_to_log += "Initializing YOLO\n"
    yolo(ns)

    # load frames into memory
    if ns.abort_flag: return
    ns.text_to_log += "Loading Reference video frame\n"
    ns.ref_data = load_frames(ns, ns.ref_path, ns.ref_rotation)
    ns.text_to_log += "Loading New video frame\n"
    if ns.abort_flag: return
    ns.video_data = load_frames(ns, ns.video_path, ns.video_rotation)

    # detect objects in each video
    if ns.abort_flag: return
    ns.text_to_log += "Finding objects in Reference video using YOLO\n"
    ref_objs = yolo_get_objects(ns,ns.ref_data)
    if ns.abort_flag: return
    ns.text_to_log += "Finding objects in New video using YOLO\n"
    video_objs = yolo_get_objects(ns,ns.video_data)
    ns.output_image = None

    # pass reference video detections to object tracker
    if ns.abort_flag: return
    ns.text_to_log += "Tracking objects in Reference video\n"
    ref_ct = CentroidTracker(ns.max_disappearing_ref)
    ref_obj_track = []
    for i in range(0, len(ns.ref_data)):
        objects = ref_ct.update([box[0] for box in ref_objs[i]])
        ref_obj_track.append(dict(objects.items()))
    ref_ct.deregister_all()
    ref_obj_lifetime = ref_ct.lifetime

    # pass second video detections to object tracker
    if ns.abort_flag: return
    ns.text_to_log += "Tracking objects in New video\n"
    video_ct = CentroidTracker(ns.max_disappearing_video)
    video_obj_track = []
    for i in range(0, len(ns.video_data)):
        objects = video_ct.update([box[0] for box in video_objs[i]])
        video_obj_track.append(dict(objects.items()))
    video_ct.deregister_all()
    video_obj_lifetime = video_ct.lifetime

    # objects are found with their full
    if ns.abort_flag: return
    mid_frames1 = [round((start[0]+end[0])/2) for objID, start, end in ref_obj_lifetime]
    mid_frames2 = [round((start[0]+end[0])/2) for objID, start, end in video_obj_lifetime]

    # matching objects in both videos 
    if ns.abort_flag: return
    ns.text_to_log += "Matching objects\n"
    if len(ref_obj_lifetime) < len(video_obj_lifetime):
        ns.obj_pairs, ns.score_history = couple_objectIDs(ns,ns.ref_data, ns.video_data, ref_obj_lifetime, video_obj_lifetime, ref_obj_track, video_obj_track, mid_frames1,mid_frames2)
    else:
        ns.obj_pairs, ns.score_history = couple_objectIDs(ns,ns.video_data, ns.ref_data, video_obj_lifetime, ref_obj_lifetime, video_obj_track, ref_obj_track, mid_frames2, mid_frames1)
        ns.obj_pairs = dict([(value, key) for key, value in ns.obj_pairs.items()])

    #
    if ns.abort_flag: return
    missing_objs = [x for x in range(0,len(ref_obj_lifetime)) if x not in ns.obj_pairs.keys()]
    ns.text_to_log += "Number of missing objects: "+len(missing_objs)+"\n"
    new_objs = [x for x in range(0,len(video_obj_lifetime)) if x not in ns.obj_pairs.values()]
    ns.text_to_log += "Number of new objects: "+len(new_objs)+"\n"

    # approximating location of missing objects
    if ns.abort_flag: return
    approx_missing = []
    if not len(missing_objs):
        ns.text_to_log += "Approximating location of missing objects\n"
    for obj in missing_objs:
        if ns.abort_flag: return
        i = 0
        while(ref_obj_lifetime[i][0] != obj):
            i += 1

        i_before_ref = 0
        while(obj != 0 and ref_obj_lifetime[i_before_ref][0] != obj-1):
            i_before_ref = +1
        i_after_ref = 0
        while(ref_obj_lifetime[i_after_ref][0] != obj):
            i_after_ref += 1
        i_after_ref += 1
        if i_after_ref >= len(ref_obj_lifetime):
            i_after_ref -= 1

        i_before_video = 0
        while(obj != 0 and video_obj_lifetime[i_before_video][0] != ns.obj_pairs[obj]-1):
            i_before_video += 1
        i_after_video = 0
        while(video_obj_lifetime[i_after_video][0] != ns.obj_pairs[ref_obj_lifetime[i_after_ref][0]]):
            i_after_video += 1

        approx_start_frame = math.ceil((ref_obj_lifetime[i][1][0] * video_obj_lifetime[i_before_video][2][0]) / ref_obj_lifetime[i_before_ref][2][0])
        approx_end_frame = math.floor((ref_obj_lifetime[i][2][0] * video_obj_lifetime[i_after_video][1][0]) / ref_obj_lifetime[i_after_ref][1][0])
        ref_step_incr = (ref_obj_lifetime[i][2][0] - ref_obj_lifetime[i][1][0]) / (approx_end_frame - approx_start_frame)

        approx_missing.append([obj, approx_start_frame, approx_end_frame, ref_obj_lifetime[i][1][0], ref_obj_lifetime[i][2][0], ref_step_incr])
    
    if ns.abort_flag: return
    output_frames = copy.deepcopy(ns.video_data)
    
    for mobj in approx_missing:
        video_ptr = mobj[1]
        ref_ptr = mobj[1]
        while(video_ptr < mobj[2]):
            if mobj[0] in ref_obj_track[math.floor(ref_ptr)].keys():
                if ns.abort_flag: return
                left, top, width, height = ref_obj_track[round(ref_ptr)][mobj[0]][1]
                cv2.rectangle(output_frames[video_ptr], (left, top), (left + width,top + height),(0, 0, 255),2)
                cv2.putText(output_frames[video_ptr], "Missing", (left, top), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.5, (255,0,0), 1, cv2.LINE_AA)
            ref_ptr+=mobj[5]
            video_ptr+=1
    
    if ns.abort_flag: return
    ns.text_to_log += "Displaying results\n"
    for i in range(0,len(video_obj_track)):
        for objID in video_obj_track[i].keys():
            if objID in new_objs:
                left, top, width, height = video_obj_track[i][objID][1]
                cv2.rectangle(output_frames[i], (left, top), (left + width,top + height),(0, 255, 0),2)
                cv2.putText(output_frames[i], "New", (left, top), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.5, (255,0,0), 1, cv2.LINE_AA)
                ns.output_image = output_frames[i]
    ns.output_image = None
    ns.running_flag = False



def yolo(ns):
    global net, outputlayers, classes

    # load weights and configuration files
    net = cv2.dnn.readNet(ns.yolo_weights, ns.yolo_config)

    # set computation backend
    net.setPreferableBackend(ns.computation_backend)

    # set target compurting device
    net.setPreferableTarget(ns.target_device)

    # define output layers
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    # load object classes
    with open(ns.yolo_classes, "r") as f:
        classes = [line.strip() for line in f.readlines()]


def initUI(ns, l):
    # Create main window
    root = Tk()

    # set title of window
    root.title("Change Detection")

    # set size and position of window ("window width x window height + position right + position down")
    root.geometry(ns.window_size)

    # disable resizing of window for user
    # root_frame.resizable(False, False)

    root_frame = Frame(root, width=670, height=520, bd=5)

    def browse_video_file(path_entry):
        """Gets the file path and sets the correspoing Entry widget with the file path"""
        # show file picker
        root_frame.filename = filedialog.askopenfilename(initialdir = ".",title = "Browse",filetypes = (("avi files","*.avi"),("mp4 files","*.mp4"),("flv files","*.flv"),("all files","*.*")))
        
        # remove text in Entry widget
        path_entry.delete(0, END)

        # set text in Entry widget to file path
        path_entry.insert(0, root_frame.filename)
    
    def set_ref_rotation(*args):
        """set values for ref_rotation global variable from dropdown"""
        option = ref_rotation_option.get()
        if option == "Rotation-90":
            ns.ref_rotation = cv2.ROTATE_90_CLOCKWISE
        elif option == "Rotation-180":
            ns.ref_rotation = cv2.ROTATE_180
        elif option == "Rotation-270":
            ns.ref_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            ns.ref_rotation = None

    def set_video_rotation(*args):
        """set values for video_rotation global variable from dropdown"""
        option = video_rotation_option.get()
        if option == "Rotation-90":
            ns.video_rotation = cv2.ROTATE_90_CLOCKWISE
        elif option == "Rotation-180":
            ns.video_rotation = cv2.ROTATE_180
        elif option == "Rotation-270":
            ns.video_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            ns.video_rotation = None

    def set_computation_backend(*args):
        """set values for computation_backend global variable from dropdown"""
        option = computation_backend_option.get()
        if option == "Default":
            ns.computation_backend = cv2.dnn.DNN_BACKEND_DEFAULT
        elif option == "OpenCV":
            ns.computation_backend = cv2.dnn.DNN_BACKEND_OPENCV
        elif option == "Halide":
            ns.computation_backend = cv2.dnn.DNN_BACKEND_HALIDE
        elif option == "Inference Engine":
            ns.computation_backend = cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE
        elif option == "VKCOM":
            ns.computation_backend = cv2.dnn.DNN_BACKEND_VKCOM
        elif option == "CUDA":
            ns.computation_backend = cv2.dnn.DNN_BACKEND_CUDA

    def set_target_device(*args):
        """set values for target_device global variable from dropdown"""
        option = target_device_option.get()
        if option == "CPU":
            ns.target_device = cv2.dnn.DNN_TARGET_CPU
        elif option == "OPENCL":
            ns.target_device = cv2.dnn.DNN_TARGET_OPENCL
        elif option == "OPENCL_FP16":
            ns.target_device = cv2.dnn.DNN_TARGET_OPENCL_FP16
        elif option == "MYRIAD":
            ns.target_device = cv2.dnn.DNN_TARGET_MYRIAD
        elif option == "VULKAN":
            ns.target_device = cv2.dnn.DNN_TARGET_VULKAN
        elif option == "FPGA":
            ns.target_device = cv2.dnn.DNN_TARGET_FPGA
        elif option == "CUDA":
            ns.target_device = cv2.dnn.DNN_TARGET_CUDA
        elif option == "CUDA_FP16":
            ns.target_device = cv2.dnn.DNN_TARGET_CUDA_FP16
        print(ns)
        
    def abort_script():
        MsgBox = tkMessageBox.askquestion ('Abort','Are you sure you want to abort',icon = 'warning')
        if MsgBox == "yes":
            ns.abort_flag = True
    
    def exit_script():
        MsgBox = tkMessageBox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
        if MsgBox == "yes":
            root.destroy()

    def validate_options():
        ns.ref_path = ref_dir_path_entry.get()
        if not (os.path.exists(ref_dir_path_entry.get()) and os.access(ns.ref_path, os.R_OK)):
            tkMessageBox.showinfo("Reference video", "Reference video does not exist at the specified path or the file cannot be accessed")
            return

        ns.video_path = video_dir_path_entry.get()
        if not (os.path.exists(video_dir_path_entry.get()) and os.access(ns.video_path, os.R_OK)):
            tkMessageBox.showinfo("New video", "New video does not exist at the specified path or the file cannot be accessed")
            return

        try:
            conf_thresh = float(conf_threshold_entry.get())
            if not (conf_thresh >= 0 and conf_thresh <= 1):
                raise Exception("Value not in range")
            ns.CONF_THRESHOLD = conf_thresh
        except:
            tkMessageBox.showinfo("CONF_THRESHOLD", "Value of CONF_THRESHOLD must be between 0 and 1")
            return

        try:
            nms_thresh = float(nms_threshold_entry.get())
            if not (nms_thresh >= 0 and nms_thresh <= 1):
                raise Exception("Value not in range")
            ns.NMS_THRESHOLD = nms_thresh
        except:
            tkMessageBox.showinfo("NMS_THRESHOLD", "Value of NMS_THRESHOLD must be between 0 and 1")
            return

        try:
            max_dissapeared_ref = int(max_disappeared_ref_entry.get())
            if not max_dissapeared_ref >= 0:
                raise Exception("Value not in range")
        except:
            tkMessageBox.showinfo("max_disappeared_ref", "Value of max_disappeared must be an integer greater than or equal to 0")
            return

        try:
            max_dissapeared_video = int(max_disappeared_video_entry.get())
            if not max_dissapeared_video >= 0:
                raise Exception("Value not in range")
        except:
            tkMessageBox.showinfo("max_disappeared_video", "Value of max_disappeared must be an integer greater than or equal to 0")
            return

        ref_dir_path_entry.config(state='disabled')
        ref_browse_button.config(state='disabled')
        ref_rotation_dropdown.config(state='disabled')
        video_dir_path_entry.config(state='disabled')
        video_browse_button.config(state='disabled')
        video_rotation_dropdown.config(state='disabled')
        computation_backend_dropdown.config(state='disabled')
        target_device_dropdown.config(state='disabled')
        conf_threshold_entry.config(state='disabled')
        nms_threshold_entry.config(state='disabled')
        max_disappeared_ref_entry.config(state='disabled')
        max_disappeared_video_entry.config(state='disabled')
        exit_button.config(state='disabled')
        ns.running_flag = True

        CP = Process(target=detect_changes, args=(ns,l))
        CP.start()

    # list of values for rotation dropdown menu
    rotation_choices = ["Roatation-None", "Rotation-90", "Rotation-180", "Rotation-270"]

    # first row of widgets
    ref_dir_label = Label(root_frame, text="Reference video")    # text label
    ref_dir_path_entry = Entry(root_frame, width=50, borderwidth=3)   # Entry widget for file path
    ref_browse_button = Button(root_frame, text="Browse", command=lambda: browse_video_file(ref_dir_path_entry))  # Browse button
    ref_rotation_option = StringVar(root_frame)   # variable to store option selected in dropdown menu
    ref_rotation_option.set(rotation_choices[0])    # default option in dropdown menu
    ref_rotation_dropdown = OptionMenu(root_frame, ref_rotation_option, *rotation_choices)    # dropdown menu
    ref_rotation_dropdown.config(width=12)  # width of dropdown menu
    ref_rotation_option.trace("w", set_ref_rotation)    # callback function is called when value is changed in menu

    # second row of widgets
    video_dir_label = Label(root_frame, text="New video")
    video_dir_path_entry = Entry(root_frame, width=50, borderwidth=3)
    video_browse_button = Button(root_frame, text="Browse", command=lambda: browse_video_file(video_dir_path_entry))
    video_rotation_option = StringVar(root_frame)
    video_rotation_option.set(rotation_choices[0])
    video_rotation_dropdown = OptionMenu(root_frame, video_rotation_option, *rotation_choices)
    video_rotation_dropdown.config(width=12)
    video_rotation_option.trace("w", set_video_rotation)
    
    # third row of widgets
    computation_backend_label = Label(root_frame, text="Computation_backend")
    computation_backend_choices = ["Default", "OpenCV", "Halide", "Inference Engine", "VKCOM", "CUDA"]
    computation_backend_option = StringVar(root_frame)
    computation_backend_option.set(computation_backend_choices[0])
    computation_backend_dropdown = OptionMenu(root_frame, computation_backend_option, *computation_backend_choices)
    computation_backend_dropdown.config(width=15)
    computation_backend_option.trace("w", set_computation_backend)

    # fourth row of widgets
    target_device_label = Label(root_frame, text="Target device")
    target_device_choices = ["CPU", "OPENCL", "OPENCL_FP16", "MYRIAD", "VULKAN", "FPGA", "CUDA", "CUDA_FP16"]
    target_device_option = StringVar(root_frame)
    target_device_option.set(target_device_choices[0])
    target_device_dropdown = OptionMenu(root_frame, target_device_option, *target_device_choices)
    target_device_dropdown.config(width=15)
    target_device_option.trace("w", set_target_device)

    # fifth row of widgets
    conf_threshold_label = Label(root_frame, text="CONF_THRESHOLD")
    conf_threshold_entry = Entry(root_frame, width=6, borderwidth=3)
    conf_threshold_entry.insert(0, "0.7")   # set default value

    # sixth row of widgets
    nms_threshold_label = Label(root_frame, text="NMS_THRESHOLD")
    nms_threshold_entry = Entry(root_frame, width=6, borderwidth=3)
    nms_threshold_entry.insert(0, "0.4")

    # seventh row of widgets
    max_disappeared_ref_label = Label(root_frame, text="max_disappearing\nfor reference\nvideo")
    max_disappeared_ref_entry = Entry(root_frame, width=6, borderwidth=3)
    max_disappeared_ref_entry.insert(0, "10")   # set default value

    # eighth row of widgets
    max_disappeared_video_label = Label(root_frame, text="max_disappearing\nfor new\nvideo")
    max_disappeared_video_entry = Entry(root_frame, width=6, borderwidth=3)
    max_disappeared_video_entry.insert(0, "10")

    # add first row of widgets to grid
    ref_dir_label.grid(row=0, column=0, sticky=W)
    ref_dir_path_entry.grid(row=0, column=1)
    ref_browse_button.grid(row=0, column=2)
    ref_rotation_dropdown.grid(row=0, column=3)

    # add second row of widgets to grid
    video_dir_label.grid(row=1, column=0, sticky=W)
    video_dir_path_entry.grid(row=1, column=1)
    video_browse_button.grid(row=1, column=2)
    video_rotation_dropdown.grid(row=1, column=3)

    # add third row of widgets to grid
    computation_backend_label.grid(row=2, column=0, sticky=W)
    computation_backend_dropdown.grid(row=2, column=1, sticky=W)

    # add fourth row of widgets to grid
    target_device_label.grid(row=3, column=0, sticky=W)
    target_device_dropdown.grid(row=3, column=1, sticky=W)

    # add fifth row of widgets to grid
    conf_threshold_label.grid(row=4, column=0, sticky=W)
    conf_threshold_entry.grid(row=4, column=1, sticky=W)

    # add sixth row of widgets to grid
    nms_threshold_label.grid(row=5, column=0, sticky=W)
    nms_threshold_entry.grid(row=5, column=1, sticky=W)

    # add seventh row of widgets to grid
    max_disappeared_ref_label.grid(row=6, column=0, sticky=W)
    max_disappeared_ref_entry.grid(row=6, column=1, sticky=W)

    # add eighth row of widgets to grid
    max_disappeared_video_label.grid(row=7, column=0, sticky=W)
    max_disappeared_video_entry.grid(row=7, column=1, sticky=W)

    # run button
    run_button = Button(root_frame, text="Run", fg="Green", command=validate_options)
    run_button.grid(row=8, column=1, sticky=E)

    # abort button
    abort_button = Button(root_frame, text="Abort", fg="Red", command=abort_script)
    abort_button.grid(row=8, column=2)

    # exit button
    exit_button = Button(root_frame, text="Exit", command=exit_script)
    exit_button.grid(row=8, column=3, sticky=W)

    # concole log
    log_frame=Frame(root_frame, bd=5, relief=RIDGE)

    yscrollbar = Scrollbar(log_frame)
    yscrollbar.pack( side = RIGHT, fill = Y )

    console_log = Text(log_frame, height=13, yscrollcommand = yscrollbar.set, bg="Black", fg="White")
    console_log.config(state="normal")
    console_log.insert(INSERT, ns.text_to_log)
    console_log.config(state="disabled")
    ns.text_to_log = ""
    console_log.pack( side = LEFT, fill = BOTH )

    yscrollbar.config( command = console_log.yview )

    log_frame.grid(row=9, columnspan=4)

    root_frame.grid(row=0, column=0)

    image_frame = Frame(root, width=600, height=520, bd=2, relief=SUNKEN)
    text_label =Label(image_frame, text="Image display area", width=60, height=34)
    if not type(ns.output_image) == type(None):
        if(ns.output_image.shape[0] > ns.output_image.shape[1]):
            ns.output_image = image_resize(ns.output_image, height = 500)
        else:
            ns.output_image = image_resize(ns.output_image, width = 500)
        ns.output_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(ns.output_image, cv2.COLOR_BGR2RGB)))
        text_label =Label(image_frame, image=ns.output_image)
    text_label.pack()
    image_frame.grid(row=0, column=1)

    if not ns.running_flag:
        ref_dir_path_entry.config(state='normal')
        ref_browse_button.config(state='normal')
        ref_rotation_dropdown.config(state='normal')
        video_dir_path_entry.config(state='normal')
        video_browse_button.config(state='normal')
        video_rotation_dropdown.config(state='normal')
        computation_backend_dropdown.config(state='normal')
        target_device_dropdown.config(state='normal')
        conf_threshold_entry.config(state='normal')
        nms_threshold_entry.config(state='normal')
        max_disappeared_ref_entry.config(state='normal')
        max_disappeared_video_entry.config(state='normal')
        exit_button.config(state='normal')

    root.mainloop()

if __name__ == '__main__':
    manager = Manager()
    ns = manager.Namespace()
    ns.yolo_weights = "./Yolo/yolov3.weights"
    ns.yolo_config = "./Yolo/yolov3.cfg"
    ns.yolo_classes = "./Yolo/coco.names"
    ns.CONF_THRESHOLD = 0.7
    ns.NMS_THRESHOLD = 0.4
    ns.IMG_WIDTH = 416
    ns.IMG_HEIGHT = 416
    ns.computation_backend = cv2.dnn.DNN_BACKEND_OPENCV
    ns.target_device = cv2.dnn.DNN_TARGET_OPENCL
    ns.ref_rotation = None
    ns.video_rotation = None
    ns.ref_data = []
    ns.video_data = []
    ns.ref_path = ""
    ns.video_path = ""
    ns.max_disappearing_ref = 10
    ns.max_disappearing_video = 10
    ns.obj_pairs = {}
    ns.score_history = {}
    ns.window_size = '1200x530+0+0'
    ns.abort_flag = False
    ns.running_flag = False
    ns.text_to_log = ""
    ns.output_image = None
    l = manager.list(range(10))
    UIP = Process(target = initUI, args=(ns,l))
    UIP.start()
    UIP.join()
    print(ns)
    #print(l)
