import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import argparse

# LOADING THE IMAGE
# Example usage: python hough.py -n No_entry/NoEntry0.bmp
parser = argparse.ArgumentParser(description='no entry detection')
parser.add_argument('-name', '-n', type=str, default='No_entry/NoEntry0.bmp')
args = parser.parse_args()

# /** Global variables */
cascade_name = "NoEntrycascade/cascade.xml"


def detectBox(frame):

	# Pre-processing Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    no_entry = model.detectMultiScale(frame_gray, scaleFactor=1.05, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(10,10), maxSize=(200,200))
    return no_entry

def houghSpace(frame_edge, rad=30):

    # Initialize the Hough space
    hSpace = np.zeros_like(frame_edge, dtype = np.float32)
    edgePoint = np.column_stack(np.where(frame_edge > 0))

    # vote on the hough space
    for j,k in edgePoint:
        for i in np.linspace(0, 2 * np.pi, 360):
            x = int(j - rad * np.cos(i))
            y = int(k - rad * np.sin(i))
            if 0 <= x < hSpace.shape[1] and 0 <= y < hSpace.shape[0]:
                hSpace[y, x] += 1

    hSpace = cv2.normalize(hSpace, None, 0, 255, cv2.NORM_MINMAX) 
    plt.figure(figsize=(10, 10))
    plt.title(f"Hough Space for Radius = {rad}")
    plt.imshow(hSpace, cmap='hot')
    plt.colorbar()
    plt.savefig(f"hough_space_{rad}.png")
    plt.close()

def houghCircle(frame):

    # Pre-processing the image by turinging it into grayscale, apply gaussian blur and canny for better edge detecting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (9,9), 2)
    frame_edge = cv2.Canny(frame_blur, 50, 150)

    # Saving processed image to detectEdge.jpg
    cv2.imwrite("detectEdge.jpg", frame_edge)

    # Apply Hough circle detection on the image
    circle = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 
                              dp= 1.5,
                              minDist= 30,
                              param1= 100,
                              param2= 30,
                              minRadius= 5,
                              maxRadius= 50)
    if circle is not None:
        circle = np.uint16(np.around(circle))
            
    return circle, frame_edge

def combineDetect(detectVJ, detectCircle):

    matchBox = []

    # Loop around the circle detected by Hough circle and sign detected by Viola-Jones
    if detectCircle is not None:
        for i in detectCircle[0, :]:
            x = int(i[0])
            y = int(i[1])
            r = int(i[2])
            cX = x -r
            cY = y - r
            cWidth = r * 2
            cHeight = r * 2
            if detectCircle is not None:
                for j in detectVJ:
                    vX = int(j[0])
                    vY = int(j[1])
                    vWidth = int(j[2])
                    vHeight = int(j[3])
                    x1 = max(cX, vX)
                    y1 = max(cY, vY)
                    x2 = min(cX + cWidth, vX + vWidth)
                    y2 = min(cY + cHeight, vY + vHeight)
                    intersect = max(0, x2 - x1) * max(0, y2 - y1) #area of intersected box
                    union = ((cWidth * cHeight) # area of hough circle box
                            + (vWidth * vHeight) # area of VJ box
                            - intersect)
                    iou = intersect / union if union > 0 else 0
                    if iou > 0.5: # iou threshold set as 0.5
                        matchBox.append(j)
    return matchBox

def readGroundtruth(filename='groundtruth.txt', frame=None, imageName=None):

    imageName = os.path.basename(imageName)
    truthBox = []
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
                colour = (0, 0, 255)  # Blue for ground truth boxes
                thickness = 2
                truthBox.append((x, y, width, height))
                frame = cv2.rectangle(frame, start, end, colour, thickness)
    return truthBox

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


# 3. Detect the sign with both method and display it along with the ground truth
detectVJ = detectBox( frame )
detectCircle, edgeFrame = houghCircle(frame)
matchBox = combineDetect(detectVJ, detectCircle)
print(len(matchBox))
for i in range(0, len(matchBox)):
        start_point = (matchBox[i][0], matchBox[i][1])
        end_point = (matchBox[i][0] + matchBox[i][2], matchBox[i][1] + matchBox[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
readGroundtruth('groundtruth.txt', frame, imageName)

houghSpace(edgeFrame, 30)


# 4. Save Result Image
cv2.imwrite( "detectedVJH.jpg", frame )


