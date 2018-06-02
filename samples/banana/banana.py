"""
Mask R-CNN
Train on the fruit image dataset (only banana for now) dataset

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written/Modified by Nils Dralle

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 banana.py train --dataset=/path/to/Fruit-Image-Dataset/ --weights=coco

    # Resume training a model that you had trained earlier
    python3 banana.py train --dataset=/path/to/Fruit-Image-Dataset/ --weights=last

    # Train a new model starting from ImageNet weights
    python3 banana.py train --dataset=/path/to/Fruit-Image-Dataset/ --weights=imagenet

    # Apply color splash to an image
    python3 banana.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
"""

import os
import sys
import json
import datetime
from math import fabs as abs
import numpy as np
import cv2 as cv
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Path to Fruit-Image-Dataset (adjust to your system)
FID_DIR= os.path.abspath("../../../Fruit-Images-Dataset/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BananaConfig(Config):
    """Configuration for training on the banana dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "banana"
    
    # ?
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + banana

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence (might need to be adjusted)
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class BananaDataset(utils.Dataset):
    
    def reduce_noise(self,img, background_color, per_ch_noise_threshold):
        (h,w)=img.shape[:2]
        rn=img.copy()
        for i in range(w):
            for j in range(h):
                px=img[j,i]
                d_blue=abs(background_color[0]-px[0]) 
                d_green=abs(background_color[1]-px[1]) 
                d_red=abs(background_color[2]-px[2])
                if d_red <= per_ch_noise_threshold and d_green <= per_ch_noise_threshold and d_blue <= per_ch_noise_threshold:
                    rn[j,i]=background_color
        return rn

    def calculate_bbox(self,img,background_color):
        (h,w)=img.shape[:2]
        xmin=ymin=xmax=ymax=-1
        for i in range(w):
            for j in range(h):
                px=img[j,i]
                if (background_color != px).all():
                    if xmin == -1:
                        xmin=i
                    if ymin == -1 or j < ymin:
                        ymin=j
                    if xmax < i:
                        xmax=i
                    if ymax < j:
                        ymax=j
        return (xmin,ymin,xmax,ymax)
    
    def calculate_normalized_bbox(self,img,background_color):
        (h,w)=img.shape[:2]
        bbox=self.calculate_bbox(img, background_color)
        return(bbox[0]/w,bbox[1]/h,bbox[2]/w,bbox[3]/h)
    
    def apply_bounding_box(self,img,nbbox,draw_color,thickness):
        (h,w)=img.shape[:2]
        xmin=int(nbbox[0]*w)
        ymin=int(nbbox[1]*h)
        xmax=int(nbbox[2]*w)
        ymax=int(nbbox[3]*h)
        bbox_img=img.copy()
        cv.rectangle(bbox_img,(xmin,ymin),(xmax,ymax),draw_color,thickness)
        return bbox_img
        
    def calculate_mask(self, img,background_color):
        (h,w)=img.shape[:2]
        mask=np.zeros((h,w,1))
        for i in range(w):
            for j in range(h):
                px=img[j,i]
                if (background_color == px).all():
                    mask[j,i]=0
                else:
                    mask[j,i]=1
        return mask


    def load_banana(self, dataset_dir, subset):
        """Load a subset of the Banana dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("banana", 1, "banana")

        # Train or validation dataset?
        assert subset in ["Training/Banana", "Validation/Banana"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Go through dataset adn generate relevant artifacts
        for subdir, dirs, files in os.walk(dataset_dir):
            for file in files:
                #Load original image
                original=cv.imread(os.path.join(subdir,file))
                #get dimensions
                (h, w)=original.shape[:2]
                #Assume bg color to be white (BGR)
                background_color=[255]*3
                #Red (BGR)
                red_color=[0,0,255]
                #Reduce noise on background to remove jpeg compression artifacts
                rnoise=self.reduce_noise(original,background_color,25)
                #calculate bounding box
                bbox=self.calculate_bbox(rnoise, background_color)
                #calculate normalized bounding box
                nbbox=self.calculate_normalized_bbox(rnoise, background_color)
                #apply bounding box to image for debugging
                bbox_img=self.apply_bounding_box(rnoise, nbbox, red_color, 3)
                #create image mask
                mask=self.calculate_mask(rnoise, background_color)
                #save img info to parent
                self.add_image(
                    "banana",
                    image_id=file,  # use file name as a unique image id
                    path=os.path.join(subdir,file),
                    width=w, height=h,
                    image=rnoise,
                    mask=mask)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "banana":
            return super(self.__class__, self).load_mask(image_id)

        # Convert mask image to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask=info["mask"]
        (h,w)=mask.shape[:2]
        #Only one instance per image
        extended_mask=np.zeros([h, w, 1],
                        dtype=np.uint8)
        
        for i in range(w):
            for j in range(h):
                extended_mask[j,i,0]=mask[j,i]
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return extended_mask.astype(np.bool), np.ones([extended_mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "banana":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BananaDataset()
    dataset_train.load_banana(args.dataset, "Training/Banana")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BananaDataset()
    dataset_val.load_banana(args.dataset, "Validation/Banana")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None):
    assert image_path

    # Only support images
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)    
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect bananas.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Fruit-Image-Dataset/",
                        help='Directory of the Fruit-Image-Dataset dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BananaConfig()
    else:
        class InferenceConfig(BananaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
