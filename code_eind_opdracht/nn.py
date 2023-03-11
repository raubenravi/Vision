import tensorflow as tf
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import skimage.io as io
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


images = []
masks = []

i = 1
# load image
img_path = "./data/Image/{}.jpg".format(i)
print(img_path)
img = io.imread(img_path)[:,:,:3]
#image_rescaled = rescale(img, 0.25, anti_aliasing=False)
#img = tf.image.resize(img, [256, 256] , method='nearest')
grayscale = rgb2gray(img)

grayscale = np.array(grayscale, dtype=np.float64) / 255
w, h = original_shape = tuple(grayscale.shape)
image_array = np.reshape(grayscale, (w * h, 1))

image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
kmeans = KMeans(n_clusters=5, n_init="auto", random_state=0).fit(
    image_array_sample
)


labels = kmeans.predict(image_array)




codebook_random = shuffle(image_array, random_state=0, n_samples=5)

labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)
new_image  = recreate_image(kmeans.cluster_centers_, labels, w, h)
layer = tf.keras.layers.ReLU(max_value=2.0)
#kmeans.cluster_centers_
Relu = layer(codebook_random)
Relu = recreate_image(codebook_random, labels, w, h)
Relu_array = np.reshape(Relu, (w * h, 1))

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
ax = axes.ravel()
mask_path = "./data/Mask/{}.png".format(i)
mask = tf.io.read_file(mask_path)
mask = tf.io.decode_png(mask, channels=1)
mask = tf.where(mask == 255, 1, 0)
#mask = tf.image.resize(mask, [256, 256] , method='nearest')
mask_array = np.reshape(mask, (w * h, 1))


# load mask


print(Relu_array.shape)
print(mask_array.shape)
#(x_train, y_train), (x_test, y_test) = train_test_split(Relu_array, mask_array, test_size=0.1, random_state=42)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1, 1)),
  tf.keras.layers.Dense(50)
])
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(Relu_array, mask_array, epochs=1)


i = 3
# load image
img_path = "./data/Image/{}.jpg".format(i)
print(img_path)
img = io.imread(img_path)[:,:,:3]
#image_rescaled = rescale(img, 0.25, anti_aliasing=False)
#img = tf.image.resize(img, [256, 256] , method='nearest')
grayscale = rgb2gray(img)

grayscale = np.array(grayscale, dtype=np.float64) / 255
w, h = original_shape = tuple(grayscale.shape)
image_array = np.reshape(grayscale, (w * h, 1))

image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
kmeans = KMeans(n_clusters=5, n_init="auto", random_state=0).fit(
    image_array_sample
)
labels = kmeans.predict(image_array)

codebook_random1 = shuffle(image_array, random_state=0, n_samples=5)
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
#layer = tf.keras.layers.ReLU(max_value=5.0)
Relu = layer(codebook_random)
Relu = recreate_image(codebook_random, labels, w, h)
Relu_array = np.reshape(Relu, (w * h, 1))


preds = model.predict(Relu_array)
print(Relu_array.shape)
preds = preds[ :,:1]
image  = recreate_image(preds, labels, w, h)
plt.imshow(image)
plt.show()
