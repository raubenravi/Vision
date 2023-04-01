
import matplotlib.pyplot as plt
from skimage.morphology import square
import skimage.morphology


import numpy as np
import skimage.io
from skimage.color import rgb2hsv, rgb2gray
import skimage.io as io
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import tensorflow as tf
from skimage import io, filters

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)
from skimage.morphology import disk


def splitIntoReducedColors(img, kmean_clusters = 10):

    img = np.array(img, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(img.shape)
    image_array = np.reshape(img, (w * h, -1))
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=kmean_clusters, n_init="auto", random_state=0).fit(
        image_array_sample)

    labels = kmeans.predict(image_array)
    new_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
    print(kmeans.cluster_centers_.shape)
    masks = []
    for i in range(0, kmean_clusters):
        grayFilter = rgb2gray(kmeans.cluster_centers_)
        new_gray = rgb2gray(new_image)
        masks.append(np.where(new_gray == grayFilter[i], 1,0))
    return masks



def EvulateMask(masks = []):
    high_score = 0
    best_Mask = None
    for mask in masks:
        kernel = skimage.morphology.disk(radius=5)
        eroded_mask = skimage.morphology.erosion(mask, kernel )
        w, h = original_shape = tuple(mask.shape)
        eroded_mask = np.reshape(eroded_mask, (w * h))
        score = np.count_nonzero(eroded_mask)
        if (score > high_score ):
            high_score = score
            best_Mask = mask
    return best_Mask



patches = []
labels = []

i = 5
# load image
img_path = "./data/Image/{}.jpg".format(i)

image = io.imread(img_path)#[:, :, :3]
print((image.shape))
print("image {} loaded".format(i))
EvulateMask(masks=splitIntoReducedColors(image))
i#mage1 = splitIntoReducedColors(img=image)[3]

best_mask = EvulateMask(splitIntoReducedColors(img=image))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(image)
ax[1].imshow(best_mask)
plt.show()

mask_path = "./data/Mask/{}.png".format(i)
mask = tf.io.read_file(mask_path)
mask = tf.io.decode_png(mask, channels=1)
mask = tf.where(mask == 255, 1, 0)
# mask_array = np.reshape(mask, (w * h, 1))
# patchImage = createpatch(img=image, x=250,y=190)
# patch =np.reshape(patchImage, (10*10))
mask1 = np.reshape(mask[250][190], 1)[0]
