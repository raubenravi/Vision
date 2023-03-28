from skimage import data
import matplotlib.pyplot as plt
import skimage
from skimage.feature import CENSURE
import skimage.io as io
import  numpy as np
from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2

image = data.astronaut()
original = image


image = cv2.copyMakeBorder(image, 100, 100, 100, 100, borderType=cv2.BORDER_CONSTANT)


transform = skimage.transform.AffineTransform(rotation=50 , scale=(2,2) , shear= 1.2, translation=(30, 40))


image = skimage.transform.warp(image, transform)
#image = skimage.transform.warp(image, transformTranslation)
#image = skimage.transform.warp(image, transformshear)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].set_title("tranfsormed")
ax[1].imshow(image)
plt.show()