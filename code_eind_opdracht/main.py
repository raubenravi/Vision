import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, rgb2gray
import skimage.io as io
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

image_collection = io.imread_collection('./data/Image/*.jpg', plugin='matplotlib')
#image = io.imread('./data/Image/0.jpg')
#image = image_collection[1]
image = image_collection[5]
grayscale = rgb2gray(image)
print(grayscale.shape)
n_colors = 25

grayscale = np.array(grayscale, dtype=np.float64) / 255
w, h = original_shape = tuple(grayscale.shape)
image_array = np.reshape(grayscale, (w * h, 1))

image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=0).fit(
    image_array_sample
)


labels = kmeans.predict(image_array)


codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)

labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()


ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
clustered_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
ax[2].imshow(clustered_image, cmap=plt.cm.gray)
ax[2].set_title("clustering")

fig.tight_layout()
plt.show()
patch = clustered_image[0:5,0:5]
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))
ax0.hist(clustered_image.ravel())
ax0.set_title("Histogram of the clustered_image")
ax1.hist(patch.ravel())
ax1.set_title("Histogram of the clustered patch")
plt.show()