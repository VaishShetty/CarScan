# -*- coding: utf-8 -*-
"""
Modified on Fri Jul  2 19:55:47 2021

Author: Vaishak

Descrption : This code demonstrates the object segmentation using a pre-trained model.
"""

## Import required modules/packages
import os
import numpy as np
from cv2 import cv2
from pixellib.semantic import semantic_segmentation

def load_images_from_folder(folder)->list:
    '''
    Description : Loads the images from the given directory

    Input argument(s) : folder (type = str) - Input directory path to input images

    Returns : images (type = list) - List of image
    '''
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

## Main/Driver Function
if __name__ == "__main__":

    # Load Input images
    IMAGEPATH = 'Input Images'
    COLORIMAGELIST = load_images_from_folder(IMAGEPATH)

    # Process each image
    for idx, image in enumerate(COLORIMAGELIST):

        # Segment image
        segment_image = semantic_segmentation()
        segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
        process_frame = True
        segmap, segoverlay = segment_image.segmentAsPascalvoc(image, process_frame)

        # Post-Processing on image
        gray = cv2.cvtColor(segoverlay, cv2.COLOR_BGR2GRAY)
        ret2, Mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Create a mask from the replace background pixels with the grayscale value "255"
        SegOutput = cv2.bitwise_and(image, image, mask=Mask)
        b_plane = (SegOutput[:, :, 2] == 0)
        g_plane = (SegOutput[:, :, 1] == 0)
        r_plane = (SegOutput[:, :, 0] == 0)
        black = np.where(r_plane & g_plane & b_plane)
        SegOutput[black] = (255, 255, 255)

        # Write output image
        cv2.imwrite("Output images/SegOutput"+ str(idx)+".jpg", SegOutput)
        