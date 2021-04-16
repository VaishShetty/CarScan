# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:20:56 2021

@author: Vaishak
"""


from pixellib.semantic import semantic_segmentation
import numpy as np
import cv2
import os

## Load images from the folder/direcotry 
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images



## Main Function 
if __name__ == "__main__":
        
    imagePath = 'Input Images'
    
    inputImagesColor = load_images_from_folder(imagePath)
    
    for iterVar in range(len(inputImagesColor)):
            
        segment_image = semantic_segmentation()
        segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
        segmap, segoverlay = segment_image.segmentAsPascalvoc(inputImagesColor[iterVar],process_frame = True)
            
        gray = cv2.cvtColor(segoverlay, cv2.COLOR_BGR2GRAY)
        ret2,Mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        SegOutput = cv2.bitwise_and(inputImagesColor[iterVar], inputImagesColor[iterVar], mask=Mask)
        
        black=np.where((SegOutput[:,:,0]==0) & (SegOutput[:,:,1]==0) & (SegOutput[:,:,2]==0))
        SegOutput[black]=(255,255,255)
        cv2.imwrite("Output images\SegOutput"+ str(iterVar)+".jpg", SegOutput)
        