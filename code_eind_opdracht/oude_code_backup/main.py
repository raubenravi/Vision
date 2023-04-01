from tensorflow_datasets.image_classification.colorectal_histology_test import num_classes
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

patches = []
labels = []
for i in range(3) :
    #i = 1
    # load image
    img_path = "./data/Image/{}.jpg".format(i)
    try:
        image = io.imread(img_path)[:,:,:3]
    except:
        continue
    print("image {} loaded".format(i))
    grayscale = rgb2gray(image)
    grayscale = np.array(grayscale, dtype=np.float64) / 255
    w, h = original_shape = tuple(grayscale.shape)
    #shower = io.imshow(image)
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
    #print(mask1)

    train_label_array = []
    #print(image.shape)
    #image = tf.expand_dims(image, 0)
    #mask = tf.expand_dims(mask, 0)
    # Extract patches from the image tensor
    # generate the patches
    patch_size = (32, 32)
    try:
        for i in range(patch_size[0] + 1, image.shape[0], patch_size[0]):
            for j in range(patch_size[1] + 1, image.shape[1], patch_size[1]):
                patch = image[i - int(patch_size[0] / 2):i + int(patch_size[0] / 2),
                        j - int(patch_size[1] / 2):j + int(patch_size[1] / 2)]
                if (patch.shape != (patch_size[0], patch_size[1], 1)):
                    continue
                patches.append(patch)
                label = np.reshape(mask[i][j][0], 1)[0]
                #label =  np.reshape(mask[i][j][0], (1,1))[0]
                #print(label.shape)
                labels.append(label)

    except:
        print("error with number:")
        print(i)
        pass

#labels.append(0)
patches = np.array(patches)
labels = np.array(labels)

print(patches[0].shape)
print("should be the shape")
# Train the model on the patches and labels
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(32,32,1)),
            layers.Conv2D(20, kernel_size=(5, 5)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(20, kernel_size=(3, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(num_classes, kernel_size=(3, 3)),
            #tf.keras.layers.Flatten()
         ])
#louter convelouties en dan pooling
#convevelutie padding valid  -> 5*5  (2 keer) daarna 2 keer pooling
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(patches, labels, epochs=1)

model.save('saved_model/my_model')
#eroderen
#neuraalNetwerk
#neurnetwerk met convoluties



i = 3091
# load image
img_path = "./data/Image/{}.jpg".format(i)
image = io.imread(img_path)[:,:,:3]
image = resize(image, (400,400))
original = image
image = FilterImage(image)
plt.imshow(image)
plt.show()
#image = tf.expand_dims(image, 0)
print(image.shape)

patches = []
for i in range(0, image.shape[0]-patch_size[0]):
    for j in range(0, image.shape[1]-patch_size[1]):
        patch = image[i:i+patch_size[0], j:j+patch_size[1]]
        if (patch.shape != (int(patch_size[0]), int(patch_size[1]), 1)):
            patch = np.zeros((int(patch_size[0]), int(patch_size[1]), 1))
            #print(i,j)
            #continue
        #patch = resize(patch, (32*32 , 1))
        patch = tf.image.resize(patch, (  int(patch_size[0]) , int(patch_size[1])) )
        patches.append(patch)

patches = np.array(image)
preds = model.predict(image)

#preds = preds[ :,:1]
threshold = 0.76
#preds = np.where(preds > threshold, 1, 0)
#preds = preds.reshape(preds, (450,640, -1) )
#image_shape = (w, h) # specify the shape of the original image

preds = np.reshape(preds, tuple(int(i) for i in (image.shape[0]-32,image.shape[1]-32, -1)))
#image = tf.expand_dims(image, 0)
#preds = tf.expand_dims(preds, 0)

print(preds.shape)
#preds = preds.reshape(preds, (100,100, 1) )
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

preds = np.mean(preds, axis=2)


preds = np.uint8(255 * preds[:-32,:-32])

ax[0].imshow(original[:-32,:-32,:])
ax[0].set_title("Original")
ax[1].set_title("prediction")
ax[1].imshow(preds)
plt.show()

