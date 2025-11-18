
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import random
import os
import tensorflow as tf
import numpy as np
import random as r
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from keras import backend as keras
import os
from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras import backend as K

sm.set_framework('tf.keras')
sm.framework()
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
BACKBONE = 'resnet34'  
preprocess_input = sm.get_preprocessing(BACKBONE)
import cv2
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

n_classes = 1
activation = 'sigmoid'  

def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union

image_dataset = ["images\\*.jpg", "masks\\*.jpg"]

image_all = []
maskimages = []
grey_list = []


for path in image_dataset:
    image_files = glob.iglob(path)

    count = 0  

    for file in image_files:
        if count >6:
            break
        image = cv2.imread(file)
        #print("shape of images",image.shape)
        #cv2.imshow("Image", image)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        #print("shape of gray image",gray.shape)
        #cv2.imshow('gray image',gray)
        cv2.waitKey(100)

        resize_image=cv2.resize(gray,(512,512))
        #print("shape of resize image",resize_image.shape)
        #cv2.imshow('resize image',resize_image)
        cv2.waitKey(100)
        


        if "mask" in path.lower():
            maskimages.append(resize_image)
        else:
            image_all.append(resize_image)

        count += 1


print("length of mask list",len(maskimages))
print("length of images ",len(image_all))

X = np.array(image_all).reshape(-1, 512, 512, 1)
#X.astype('float32')
Y = np.array(maskimages).reshape(-1, 512, 512, 1)
#X = X / 255.0
X = np.repeat(X, 3, axis=-1)  
Y = Y.astype('float32') / 255.0
X = preprocess_input(X)
print(X.shape)

# First, split into train+val and test
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Now split train+val into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)



model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=(512, 512, 3),
    classes=1,
    activation=activation,
    encoder_weights='imagenet'
)



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy',dice_metric])

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=4
)


results = model.evaluate(X_test, Y_test, batch_size=4)
print("Loss:", results[0])
print(" Accuracy:", results[1])
print("Dice Coefficient:", results[2])

preds = model.predict(X_test)
preds_binary = (preds > 0.5).astype(np.float32)

for i in range(min(4, len(X_test))):
    plt.figure(figsize=(12,4))
    
    plt.subplot(2, 2, 1)
    plt.title('Input Image')
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(Y_test[i].squeeze(), cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title('Predicted Mask')
    plt.imshow(preds_binary[i].squeeze(), cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Boundary Box')
    
    img = X_test[i].squeeze()
    if len(img.shape) == 2: 
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img
    
    
    mask = preds_binary[i].squeeze()
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_with_boxes = img_rgb.copy()
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2) 
    
    plt.imshow(img_with_boxes)
    plt.imshow(preds_binary[i].squeeze(), alpha=0.8)

    plt.tight_layout()
    plt.show()

