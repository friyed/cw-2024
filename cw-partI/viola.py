import numpy as np
import cv2
import os
import sys
import argparse

# LOADING THE IMAGE
# Example usage: python viola.py -n No_entry/NoEntry0.png
parser = argparse.ArgumentParser(description='no entry detection')
parser.add_argument('-name', '-n', type=str, default='No_entry/NoEntry0.bmp')
args = parser.parse_args()

# /** Global variables */
cascade_name = "NoEntrycascade/cascade.xml"


def detectAndDisplay(frame):

	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    no_entry = model.detectMultiScale(frame_gray, scaleFactor=1.05, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(10,10), maxSize=(200,200))
    # 3. Print number of no entry sign found
    print(len(no_entry))
    # 4. Draw box around the sign found
    for i in range(0, len(no_entry)):
        start_point = (no_entry[i][0], no_entry[i][1])
        end_point = (no_entry[i][0] + no_entry[i][2], no_entry[i][1] + no_entry[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

def readGroundtruth(filename='groundtruth.txt', frame=None, imageName=None):

    imageName = os.path.basename(imageName)
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.strip().split()
            img_name = content_list[0]
            x = int(content_list[1])
            y = int(content_list[2])
            width = int(content_list[3])
            height = int(content_list[4])
            if img_name == imageName:
                start = (x, y)
                end = (x + width, y + height)
                colour = (0, 0, 255)  # Red for ground truth boxes
                thickness = 2
                frame = cv2.rectangle(frame, start, end, colour, thickness)

# ==== MAIN ==============================================

imageName = args.name

# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

# 1. Read Input Image
frame = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)


# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier(cascade_name)
if not model.load(cascade_name): 
    print('--(!)Error loading cascade model')
    exit(0)


# 3. Detect Faces and Display Result
detectAndDisplay( frame )
readGroundtruth('groundtruth.txt', frame, imageName)

# 4. Save Result Image
cv2.imwrite( "detected.jpg", frame )


