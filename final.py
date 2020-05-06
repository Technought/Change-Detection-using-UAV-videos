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
from skimage.measure import compare_ssim
from scipy.spatial import distance as dist
from collections import OrderedDict
from tkinter import *
from tkinter import filedialog

# Global Variables
yolo_weights = "./Yolo/yolov3.weights"
yolo_config = "./Yolo/yolov3.cfg"
yolo_classes = "./Yolo/coco.names"
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416
computation_backend = cv2.dnn.DNN_BACKEND_OPENCV
target_device = cv2.dnn.DNN_TARGET_OPENCL
video_rotation = None
ref_data = []
video_data = []
ref_dir = ""
video_dir = ""
max_disappearing_ref = 0
max_disappearing_video = 0
obj_pairs = {}
score_history = {}

# ********************************************** Change Detection **********************************************


# convert video to frames
def video_to_frames(file, out_dir):

    # load video
    cap = cv2.VideoCapture(file)

    # check if output directory exists else create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    while (cap.isOpened()):
        counter = 1
        # load each frame
        ret, frame = cap.read()

        # perform rotation if necessary
        if(type(video_rotation) != type(None)):
            frame = cv2.rotate(frame, video_rotation)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # exit loop if there are no more frame to load
        if ret == 0:
            break

        # generate file path
        FrameNo = out_dir+str(counter)+'.jpg'

        # write frame to out_dir
        cv2.imwrite(FrameNo, frame)

        counter += 1
    cap.release()


def load_frames(dir):
    data = []
    i = 1
    while i < len(os.listdir(dir)):
        img = cv2.imread(os.path.join(dir, 'Frame'+str(i)+'.jpg'))
        img = cv2.resize(img, (720, 1280), interpolation=cv2.INTER_AREA)
        data.append(img)
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


def yolo_get_objects(data):
    objs = []
    for image in data:
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(outputlayers)

        # Remove the bounding boxes with low confidence
        objs.append(list(post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)))
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


def couple_objectIDs(data1, data2, lifetime1, lifetime2, track1, track2, mid_frames1, mid_frames2):
    obj_pairs = {}
    score_history = {}
    for i in range(0, len(lifetime1)):
        score_history[i] = OrderedDict()
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
                    score_history[i][j] = score

    def get_key(dict, val):
        for key, value in dict.items():
            if val == value:
                return key

    def assign_objs(objid, ignore_list):
        ignore_list.sort()
        matched_objects = list(score_history[objid].keys())
        matched_objects.sort()
        if ignore_list == matched_objects:
            return
        ignored_score_history = copy.deepcopy(score_history)
        for i in ignore_list:
            del ignored_score_history[objid][i]
        obj_with_max_score = max(ignored_score_history[objid], key=ignored_score_history[objid].get)
        if(obj_with_max_score not in obj_pairs.values()):
            obj_pairs[objid] = obj_with_max_score
        else:
            initial_assignment = get_key(obj_pairs, obj_with_max_score)
            if score_history[initial_assignment][obj_with_max_score] > score_history[objid][obj_with_max_score]:
                ignore_list.append(obj_with_max_score)
                assign_objs(objid, ignore_list)
            else:
                obj_pairs[objid] = obj_with_max_score
                ignore_list = [obj_with_max_score]
                del obj_pairs[initial_assignment]
                assign_objs(initial_assignment, ignore_list)

    for i in score_history.keys():
        assign_objs(i, [])
    return (obj_pairs, score_history)


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


def change_detection():
    # pass reference video detections to object tracker
    ref_ct = CentroidTracker(max_disappearing_ref)
    ref_obj_track = []
    for i in range(0, len(ref_data)):
        objects = ref_ct.update([box[0] for box in ref_objs[i]])
        ref_obj_track.append(dict(objects.items()))
    ref_ct.deregister_all()
    ref_obj_lifetime = ref_ct.lifetime

    # pass second video detections to object tracker
    video_ct = CentroidTracker(20)
    video_obj_track = []
    for i in range(0, len(video_data)):
        objects = video_ct.update([box[0] for box in video_objs[i]])
        video_obj_track.append(dict(objects.items()))
    video_ct.deregister_all()
    video_obj_lifetime = video_ct.lifetime

    # objects are found with their full
    mid_frames1 = [round((start[0]+end[0])/2) for objID, start, end in ref_obj_lifetime]
    mid_frames2 = [round((start[0]+end[0])/2) for objID, start, end in video_obj_lifetime]

    #
    if len(ref_obj_lifetime) < len(video_obj_lifetime):
        obj_pairs, score_history = couple_objectIDs(ref_data, video_data, ref_obj_lifetime, video_obj_lifetime, ref_obj_track, video_obj_track, mid_frames1,mid_frames2)
    else:
        obj_pairs, score_history = couple_objectIDs(video_data, ref_data, video_obj_lifetime, ref_obj_lifetime, video_obj_track, ref_obj_track, mid_frames2, mid_frames1)
        obj_pairs = dict([(value, key) for key, value in obj_pairs.items()])

    #
    missing_objs = [x for x in range(0,len(ref_obj_lifetime)) if x not in obj_pairs.keys()]
    new_objs = [x for x in range(0,len(video_obj_lifetime)) if x not in obj_pairs.values()]

    #
    approx_missing = []
    for obj in missing_objs:
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
        while(obj != 0 and video_obj_lifetime[i_before_video][0] != obj_pairs[obj]-1):
            i_before_video += 1
        i_after_video = 0
        while(video_obj_lifetime[i_after_video][0] != obj_pairs[ref_obj_lifetime[i_after_ref][0]]):
            i_after_video += 1

        approx_start_frame = math.ceil((ref_obj_lifetime[i][1][0] * video_obj_lifetime[i_before_video][2][0]) / ref_obj_lifetime[i_before_ref][2][0])
        approx_end_frame = math.floor((ref_obj_lifetime[i][2][0] * video_obj_lifetime[i_after_video][1][0]) / ref_obj_lifetime[i_after_ref][1][0])
        ref_step_incr = (ref_obj_lifetime[i][2][0] - ref_obj_lifetime[i][1][0]) / (approx_end_frame - approx_start_frame)

        approx_missing.append([obj, approx_start_frame, approx_end_frame, ref_obj_lifetime[i][1][0], ref_obj_lifetime[i][2][0], ref_step_incr])


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
    with open(yolo_classes, "r") as f:
        classes = [line.strip() for line in f.readlines()]


def main():
    yolo()
    root = Tk()
    root.title("Change Detection")
    root.geometry('600x600')

    root.mainloop()


if __name__ == '__main__':
    main()

# load frames into memory
ref_data = load_frames(ref_dir)
video_data = load_frames(video_dir)

# detect objects in each video
ref_objs = yolo_get_objects(ref_data)
video_objs = yolo_get_objects(video_data)
