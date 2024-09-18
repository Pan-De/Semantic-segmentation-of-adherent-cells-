import tensorflow as tf
from tensorflow.keras.models import Model
from focal_loss import BinaryFocalLoss
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from patchify import patchify, unpatchify
from smooth_tiled_predictions import predict_img_with_smooth_windowing
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from stardist.models import StarDist2D 
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage import io, measure, color
from skimage.segmentation import clear_border
import math
import imutils
import random

#load image
def load_img(img_path):
    
    image=cv2.imread(img_path)

    # Convert the image to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # If it's a color image, convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        # If it's already grayscale, leave it as is
        grayscale_image = image
    else:
        print(f"Unsupported image format: {img_path}")

    return grayscale_image


# input: bright-field image, binary output (white: cells, black: background)
def predict_large_image(img_path,loaded_model,threshold): 
    img=load_img(img_path)
    img=np.expand_dims(img,-1)

    # normalize img array to 0 to 1 
    scaler = MinMaxScaler()
    # adjust img shape. For the input of ML model: (batch size, img_width, img_heigh, 1)
    input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    # size of patches for model prediciton. Our model was trained on images with a size of 256X256
    patch_size = 256

    # Predict patch by patch with smooth blending. predicitons_smooth=3D array: shape (large image width, height,1)
    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        window_size=patch_size,
        subdivisions=2,  # Minimal amount of overlap for windowing (half overlapping). Must be an even number.
        nb_classes=1,
        pred_func=(
            lambda img_batch_subdiv: loaded_model.predict((img_batch_subdiv))
        )
    )

    # remove background noise. Keep only the pixels with a high probability of being part of a cell
    predictions_smooth[predictions_smooth<threshold]=0
    
    return predictions_smooth

# Input:a binary image. Remove small objects that might not be cells (minimal area threshold).
def count_cells(large_prediction,min_area,label_cell_model):
    count=0
    labels_predict, _ = label_cell_model.predict_instances(normalize(large_prediction))
    regions = regionprops(labels_predict)
    labels_predict=clear_border(labels_predict)
    count_before=np.unique(labels_predict)
    print("before image processing, cell counts =", len(count_before)-1)

    # Create a blank RGB image of the same shape as the labeled image
    height, width = labels_predict.shape
    label_image_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign a unique random color to each label
    for region in regions:
        if region.area < min_area:
            count=count-1
            label_image_rgb[labels_predict == region.label] =0
        else:
            color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            label_image_rgb[labels_predict == region.label] = color
    
    print("total cell count=", count)
    return count,label_image_rgb


##--------------------------------------main-----------------------------------------------------------------##
# load U-Net model for cell segmentation
load_path="D:/MDA2/reg_well 256_256 UNet2_Oct_18.h5"
loaded_model=tf.keras.models.load_model(load_path,compile=False)
img_path="D:/Images/1.tif"

large_prediction=predict_large_image(img_path,loaded_model,0.5)

# Use StarDist to identify each cell
model_star = StarDist2D.from_pretrained('2D_versatile_fluo')
count,labeled_cell=count_cells(large_prediction,200,model_star)

# plot the labeled cell image
plt.figure(figsize=(10,10))
plt.imshow(labeled_cell)