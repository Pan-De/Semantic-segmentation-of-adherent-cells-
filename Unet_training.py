import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from focal_loss import BinaryFocalLoss
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import os
import random
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import pandas as pd
from tensorflow.keras.metrics import MeanIoU

#Dice metric can be a great metric to track accuracy of semantic segmentation.
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))

    return 2*intersection / union

def binarize_preprocess(x):
    # Apply the condition
    if np.max(x) > 1.0:
        scale_x=x*(1.0/255)
    else:
        scale_x=x.copy()
    
    scale_x[scale_x > 0.5]=1.0
    scale_x[scale_x < 0.6]=0.0
  
    return scale_x

def training(train_img_dir,train_msk_dir,val_img_dir,val_msk_dir,save_path):
    # Creating Data generator for training 
    #(Generally, we only apply data augmentation to our training examples)
    image_gen_train = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True)
    mask_gen_train = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,preprocessing_function = binarize_preprocess) #Binarize the output again. 
   
    #Creating Validation Data generator (No data augmentation)
    image_gen_val = ImageDataGenerator(rescale=1./255)
    mask_gen_val = ImageDataGenerator(preprocessing_function = binarize_preprocess)

    BATCH_SIZE=50
    SEED=42

    train_img_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                        color_mode='grayscale',
                                                        directory=train_img_dir,
                                                        seed=SEED,
                                                        class_mode=None)

    train_mask_gen = mask_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                        color_mode='grayscale',
                                                        directory=train_msk_dir,
                                                        seed=SEED,
                                                        class_mode=None)

    val_img_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                    color_mode='grayscale',
                                                    directory=val_img_dir,
                                                    seed=SEED,
                                                    class_mode=None)

    val_mask_gen = mask_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                    color_mode='grayscale',
                                                    directory=val_msk_dir,
                                                    seed=SEED,
                                                    class_mode=None)

    train_generator = zip(train_img_gen, train_mask_gen)
    val_generator = zip(val_img_gen, val_mask_gen)


    input_shape =(256,256,1) #(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    #Build the model
    inputs = Input(input_shape)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = tf.keras.Model(inputs, outputs)

    num_train_imgs=len(os.listdir(os.path.join(train_img_dir,'train')))
    steps_per_epoch = int(np.ceil(num_train_imgs / float(BATCH_SIZE)))

    num_val_imgs=len(os.listdir(os.path.join(val_img_dir,'val')))
    validation_steps = int(np.ceil(num_val_imgs / float(BATCH_SIZE)))

    model.compile(optimizer=Adam(learning_rate = 1e-3), loss=BinaryFocalLoss(gamma=2), metrics=[dice_metric])
    history = model.fit(train_generator, 
                        steps_per_epoch=steps_per_epoch, 
                        verbose=1, 
                        epochs=30, 
                        validation_data=val_generator, 
                        validation_steps=validation_steps)

    #save the model
    model.save(save_path)
    return model,history

#plot the training and validation accuracy and loss at each epoch
def plot_val_acc(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_train_acc(history):
    loss = history.history['loss']
    acc = history.history['dice_metric']
    val_acc = history.history['val_dice_metric']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, acc, 'y', label='Training Dice')
    plt.plot(epochs, val_acc, 'r', label='Validation Dice')
    plt.title('Training and validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.show()

#Calculate IoU and average
def get_IoU(test_img_dir, test_msk_dir,model):
    image_gen_test = ImageDataGenerator(rescale=1./255)
    mask_gen_test = ImageDataGenerator(preprocessing_function = binarize_preprocess)

    test_img_gen = image_gen_test.flow_from_directory(batch_size=50,
                                                    color_mode='grayscale',
                                                    directory=test_img_dir,
                                                    seed=42,
                                                    class_mode=None)

    test_mask_gen = mask_gen_test.flow_from_directory(batch_size=50,
                                                    color_mode='grayscale',
                                                    directory=test_msk_dir,
                                                    seed=42,
                                                    class_mode=None)
    
    x = test_img_gen.next()
    y = test_mask_gen.next()

    IoU_values = []
    for img in range(0, x.shape[0]):
        temp_img = x[img]
        ground_truth=y[img]
        temp_img_input=np.expand_dims(temp_img, 0)
        prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
        
        IoU = MeanIoU(num_classes=2)
        IoU.update_state(ground_truth[:,:,0], prediction)
        IoU = IoU.result().numpy()
        IoU_values.append(IoU)
        
    df = pd.DataFrame(IoU_values, columns=["IoU"])
    df = df[df.IoU != 1.0]    
    mean_IoU = df.mean().values
    print("Validation dataset Mean IoU is: ", mean_IoU) 


##--------------------------------------main-----------------------------------------------------------------##
train_img_dir="E:/MDA2_ML data/train/images/"
train_msk_dir="E:/MDA2_ML data/train/masks/"
val_img_dir="E:/MDA2_ML data/val/images/"
val_msk_dir="E:/MDA2_ML data/val/masks/"
save_path="D:/MDA2/reg_well 256_256 UNet2_Oct_18.h5"

# standard training
model,history=training(train_img_dir,train_msk_dir,val_img_dir,val_msk_dir,save_path)

#Calculate IoU and average 
test_img_dir="E:/MDA2_ML data/test/images/"
test_msk_dir="E:/MDA2_ML data/test/masks/"
get_IoU(test_img_dir, test_msk_dir,model)
