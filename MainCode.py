# -*- coding: utf-8 -*-

## Import required modules/packages
from pixellib.semantic import semantic_segmentation
import numpy as np
import cv2
import os

## Load images from the folder/direcotry 
def load_images_from_folder(folder):
    '''
    Description : Loads the images from the given directory 
    
    Input argument(s) : 
    1. folder (type = str) - Input directory path to input images. 
    
    Returns :
    1. images (type = list) - List of image 
    '''
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

## Main/Driver Function 
if __name__ == "__main__":
    '''
    Description : Driver function or Entry point of the program. 
    '''
    
    # Load Input images 
    imagePath = 'Input Images'
    inputImagesColor = load_images_from_folder(imagePath)
    
    # Process each image
    for iterVar in range(len(inputImagesColor)):
        
        # Segment image 
        segment_image = semantic_segmentation()
        segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
        segmap, segoverlay = segment_image.segmentAsPascalvoc(inputImagesColor[iterVar],process_frame = True)
         
        # Post-Processing on image
        gray = cv2.cvtColor(segoverlay, cv2.COLOR_BGR2GRAY)
        ret2,Mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Create a mask from the replace background pixels with the grayscale value "255"
        SegOutput = cv2.bitwise_and(inputImagesColor[iterVar], inputImagesColor[iterVar], mask=Mask)
        black=np.where((SegOutput[:,:,0]==0) & (SegOutput[:,:,1]==0) & (SegOutput[:,:,2]==0))
        SegOutput[black]=(255,255,255)
        
        # Write output image
        cv2.imwrite("Output images\SegOutput"+ str(iterVar)+".jpg", SegOutput)
        
