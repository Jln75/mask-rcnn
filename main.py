from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
	help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=False,
	help="path to class labels file")
ap.add_argument("-i", "--video", required=False,
	help="path to input video to apply Mask R-CNN to")
args = vars(ap.parse_args())

# load the class label names from disk, one label per line
#CLASS_NAMES = open(args["labels"]).read().strip().split("\n")
# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
#hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
#COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#random.seed(42)
#random.shuffle(COLORS)

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"
	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = 30

# initialize the inference configuration
config = SimpleConfig()
# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image

#cap = cv2.VideoCapture(args["video"])
cap = cv2.VideoCapture(0)

width  = cap.get(3)
height = cap.get(4)

while cap.isOpened:

    ret, frame = cap.read()
    if ret == True:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = imutils.resize(image, width=512)
        # perform a forward pass of the network to obtain the results
        print("[INFO] making predictions with Mask R-CNN...")
        r = model.detect([image], verbose=1)[0]

        # loop over of the detected object's bounding boxes and masks
        for i in range(0, r["rois"].shape[0]):
            # extract the class ID and mask for the current detection, then
            # grab the color to visualize the mask (in BGR format)
            classID = r["class_ids"][i]
            if classID == 5:
                blank_image = np.zeros((height,width,3), np.uint8)
                mask = r["masks"][:, :, i]
                color = COLORS[classID][::-1]
                # visualize the pixel-wise mask of the object
                image = visualize.apply_mask(blank_image, mask, colorsys.hsv_to_rgb([1.0,1.0,1.0]), alpha=1.0)

                cv2.imshow("retour",image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    else: 
        break
