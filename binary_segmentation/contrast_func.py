import numpy as np
import cv2
from skimage import morphology
import argparse


def improve_contrast(source):
    img = source[:,:,0]
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))  
    img_not = cv2.bitwise_not(img)  
    se = np.ones((50,50), np.uint8)  
    wth = morphology.white_tophat(img_not, se)  
    raw_minus_topwhite = img.astype(int) - wth 
    raw_minus_topwhite = ((raw_minus_topwhite>0)*raw_minus_topwhite).astype(np.uint8)  
    img = clahe.apply(raw_minus_topwhite) 
    img = img[:,:,np.newaxis] 
    d3 = np.concatenate([img, img, img], axis=2).astype(np.uint8)

    return d3