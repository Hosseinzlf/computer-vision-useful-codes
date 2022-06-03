#With this code you can make jpg file and store them in a file
import pandas as pd
import netCDF4
import numpy as np
import cv2
from netCDF4 import Dataset
import os
import matplotlib.pyplot as plt
images_directory = '/home/hossein/data/FR_TOTAL_LTS_0/data/images_db/18800'
dirs = os.listdir(images_directory)

# nc = Dataset('/home/hossein/data/FR_TOTAL_LTS_0/data/images_db/18766/1621418460.nc')
for d in dirs[100:150]:
    new_d = os.path.join(images_directory,d)
    #l is get the name of the image file that is timestamp
    l = d[:-4]
    nc = Dataset(new_d)
    hdr_var = nc.variables['raw_hdr']
    factor = hdr_var.scale_factor
    offset = hdr_var.add_offset
    netCDF4.Variable.set_auto_maskandscale(hdr_var, False) 	# Deactivate _fillValue and _scale
    hdr = np.squeeze(hdr_var[:])						# hdr is 'uint16' at this stage
    # Close the netCDF file


    # Convert to RGB image (demosaicing), correct image and get the mean value for each channel
    HRGB = np.double(cv2.cvtColor(hdr, cv2.COLOR_BayerBG2BGR))
    HRGB = np.exp(factor * HRGB + offset)
    MeanRGB = cv2.mean(HRGB)

    # Convert to gray and get the mean value
    Gray = np.double(cv2.cvtColor(hdr, cv2.COLOR_BayerBG2GRAY))
    Gray = np.exp(factor * Gray + offset)
    MeanGray = cv2.mean(Gray)

    # White balance
    B = HRGB[:, :, 0] * MeanGray[0] / MeanRGB[0]
    G = HRGB[:, :, 1] * MeanGray[0] / MeanRGB[1]
    R = HRGB[:, :, 2] * MeanGray[0] / MeanRGB[2]
    HRGBwb = cv2.merge((B,G,R))

    # Generate the tonemap image
    #ToneMap = cv2.createTonemapReinhard(gamma=0.8, intensity=+1, light_adapt=0.7)
    ToneMap = cv2.createTonemapReinhard(gamma=0.7, intensity=+2, light_adapt=0.6)
    ToneHRGBwb = ToneMap.process(np.float32(HRGBwb))
    RGB = np.clip(ToneHRGBwb*255, 0, 255).astype('uint8')

    # Apply a Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Bc = clahe.apply(RGB[:, :, 0])
    Gc = clahe.apply(RGB[:, :, 1])
    Rc = clahe.apply(RGB[:, :, 2])
    RGBc = cv2.merge((Bc, Gc, Rc))

    cv2.imwrite("/home/hossein/hcodes/images/image_{}.jpg".format(l),RGBc)    
