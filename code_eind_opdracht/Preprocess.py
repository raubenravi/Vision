import matplotlib.pyplot as plt
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

def FilterImage(img = None, grayscale = True, Relu = True, Kmean=True, n_colors = 200):
    grayscale = rgb2gray(img)
    grayscale = filters.butterworth(grayscale)

    grayscale = np.array(grayscale, dtype=np.float64) / 255
    w, h = original_shape = tuple(grayscale.shape)
    image_array = np.reshape(grayscale, (w * h, 1))

    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=125, n_init="auto", random_state=0).fit(
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

    return Relu

i = 1
##load image
img_path = "./data/Image/{}.jpg".format(i)
print(img_path)
image = io.imread(img_path)[:,:,:3]
image = FilterImage(img=image)
shower = skimage.io.imshow(image, cmap=plt.cm.gray)
plt.show()