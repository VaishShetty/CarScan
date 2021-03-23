# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 17:51:25 2021

@author: Vaishak 
"""

## Import required libraries
import cv2
import os
import numpy as np

## Load images from the folder/direcotry 
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

## Determine the lightining Conditions for image 
def determine_lighting_condition(image):
    
    #For testing purpose - uncomment below line 
    #image = cv2.add(image, np.array([50.0]))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
    dark_part = cv2.inRange(gray, 0, 50)
    bright_part = cv2.inRange(gray, 200, 255)
    
    bright_thres = 0.4
    dark_thres = 0.4
    
    total_pixel = np.size(gray)
    dark_pixel = np.sum(dark_part > 0)
    bright_pixel = np.sum(bright_part > 0)
    
    print("dark_pixel ratio", dark_pixel/total_pixel)
    print("bright_pixel ratio", bright_pixel/total_pixel)
        
    if dark_pixel/total_pixel > bright_thres:
        print("Image is underexposed!")
    if bright_pixel/total_pixel > dark_thres:
        print("Image is overexposed!")
        
    return image

def background_removal(image):

    mask = np.zeros(image.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = (300,400,4030,3020)
    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = image*mask2[:,:,np.newaxis]
    
    
    return image


## Main Function 
if __name__ == "__main__":
        
    imagePath = 'Input Images'
    
    inputImagesColor = load_images_from_folder(imagePath)
    
    for iterVar in range(len(inputImagesColor)):
            
#        image = inputImagesColor[0]
    #    image = cv2.resize(inputImagesColor[0], (1512,1216))
        
        #Determine Lighting condition
    #    LightImage = determine_lighting_condition(inputImagesColor[0])
        
        # Background removal 
        fgMask = background_removal(inputImagesColor[iterVar])
        
#        invertedImage = cv2.subtract(np.array([255.0]), fgMask)
        
    #    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #    cv2.namedWindow('LightImage', cv2.WINDOW_NORMAL)
#        cv2.namedWindow('FG Mask', cv2.WINDOW_NORMAL)
#        cv2.namedWindow('invertedImage', cv2.WINDOW_NORMAL)
        
    #    cv2.imshow('image',inputImagesColor[0]) 
    #    cv2.imshow('LightImage', LightImage)
#        cv2.imshow('FG Mask', fgMask)
#        cv2.imshow('invertedImage', invertedImage)
#        cv2.imwrite("OutputImage"+str(iterVar)+".png", invertedImage)
        cv2.imwrite("MaskImage"+str(iterVar)+".png", fgMask)
        
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()