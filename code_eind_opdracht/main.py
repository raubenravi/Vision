from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2

from Preprocess import FilterImage, recreate_image
from skimage import data
import matplotlib.pyplot as plt
import skimage.io as io
import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
from skimage.util import view_as_windows
from skimage.transform import rescale, resize, downscale_local_mean

from tensorflow_examples.profiling.resnet_model import layers

i = 1
# load image
img_path = "./data/Image/{}.jpg".format(i)
image = io.imread(img_path)[:,:,:3]
grayscale = rgb2gray(image)
grayscale = np.array(grayscale, dtype=np.float64) / 255
w, h = original_shape = tuple(grayscale.shape)
shower = io.imshow(image)
#plt.show()
image = FilterImage(img=image)
#shower = io.imshow(image)
#plt.show()

mask_path = "./data/Mask/{}.png".format(i)
mask = tf.io.read_file(mask_path)
mask = tf.io.decode_png(mask, channels=1)
mask = tf.where(mask == 255, 1, 0)
#mask_array = np.reshape(mask, (w * h, 1))
#patchImage = createpatch(img=image, x=250,y=190)
#patch =np.reshape(patchImage, (10*10))
mask1 = np.reshape(mask[250][190], 1)[0]
print(mask1)

train_label_array = []
print(image.shape)
#image = tf.expand_dims(image, 0)
#mask = tf.expand_dims(mask, 0)
# Extract patches from the image tensor
# generate the patches
patch_size = (32, 32)
patches = []
labels = []
for i in range(0, image.shape[0], patch_size[0]):
    for j in range(0, image.shape[1], patch_size[1]):
        patch = image[i:i+patch_size[0], j:j+patch_size[1]]
        if (patch.shape != (32, 32, 1)):
            continue
       # patch = resize(patch, (32*32 , 1))
        #print(patch.shape)
        #patch = tf.image.resize(patch, (32 , 32))
        patches.append(patch)
        label = np.reshape(mask[i][j][0], 1)[0]
        #print(label)
        labels.append(label)


patches = np.array(patches)
labels = np.array(labels)

# Train the model on the patches and labels
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(5, 5, input_shape=(32,32, 1)),
    tf.keras.layers.Conv2D(5, 5, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(patches, labels, epochs=10)






i = 2
# load image
img_path = "./data/Image/{}.jpg".format(i)
image = io.imread(img_path)[:,:,:3]
original = image
image = FilterImage(image)
#image = tf.expand_dims(image, 0)
print(image.shape)
patches = []
for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        patch = image[i:i+patch_size[0], j:j+patch_size[1]]
        if (patch.shape != (32, 32, 1)):
            np.zeros((32, 32, 1))
            #print(i,j)
            #continue
        #patch = resize(patch, (32*32 , 1))
        #print(patch.shape)
        patch = tf.image.resize(patch, (32 , 32))
        patches.append(patch)


patches = np.array(patches)
preds = model.predict(patches)

#preds = preds[ :,:1]
threshold = 0.76
#preds = np.where(preds > threshold, 1, 0)
#preds = preds.reshape(preds, (450,640, -1) )
#image_shape = (w, h) # specify the shape of the original image

preds = np.reshape(preds, tuple(int(i) for i in (image.shape[0],image.shape[1], -1)))
#image = tf.expand_dims(image, 0)
#preds = tf.expand_dims(preds, 0)

print(preds.shape)
#preds = preds.reshape(preds, (100,100, 1) )
plt.imshow(original)
plt.show()
plt.imshow(preds[ :,:,:1])
plt.show()