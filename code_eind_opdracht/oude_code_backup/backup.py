import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage.color import rgb2hsv, rgb2gray
import skimage.io as io
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import tensorflow as tf

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

def FilterImage(img = None, grayscale = True, Relu = True, Kmean=True, n_colors = 10):
    grayscale = rgb2gray(img)

    grayscale = np.array(grayscale, dtype=np.float64) / 255
    w, h ,d= original_shape = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=5, n_init="auto", random_state=0).fit(
        image_array_sample
    )

    labels = kmeans.predict(image_array)

    codebook_random = shuffle(image_array, random_state=0, n_samples=5)

    labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)


    layer = tf.keras.layers.ReLU(max_value=100.0)
    # kmeans.cluster_centers_
    #Relu = layer(labels_random)
    #Relu_array = np.reshape(Relu, (w * h, 1))
    Relu = recreate_image(codebook_random, labels_random, w, h)
    return rgb2gray(Relu)

i = 1
# load image
img_path = "./data/Image/{}.jpg".format(i)
print(img_path)
image = io.imread(img_path)[:,:,:3]
image = FilterImage(img=image)
shower = skimage.io.imshow(image, cmap=plt.cm.gray)
plt.show()















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
for i in range(5) :
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
    except:
        print(i)
        pass

#labels.append(0)
patches = np.array(patches)
labels = np.array(labels)

# Train the model on the patches and labels
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(5, 5, input_shape=(32,32, 1),  activation='relu'),
    tf.keras.layers.Dense(30),
    tf.keras.layers.Conv2D(5, 5, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
])

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(patches, labels, epochs=5)






i = 3018
# load image
img_path = "./data/Image/{}.jpg".format(i)
image = io.imread(img_path)[:,:,:3]
image = resize(image, (250,250))
original = image
image = FilterImage(image)
#image = tf.expand_dims(image, 0)
print(image.shape)
patches = []
for i in range(0, image.shape[0]-32):
    for j in range(0, image.shape[1]-32):
        patch = image[i:i+patch_size[0], j:j+patch_size[1]]
        if (patch.shape != (32, 32, 1)):
            patch = np.zeros((32, 32, 1))
            #print(i,j)
            #continue
        #patch = resize(patch, (32*32 , 1))
        patch = tf.image.resize(patch, (32 , 32))
        patches.append(patch)


patches = np.array(patches)
preds = model.predict(patches)

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

