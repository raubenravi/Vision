import tensorflow as tf
from skimage.color import rgb2gray
from skimage.color import rgb2hsv, rgb2gray
import skimage.io as io
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import tensorflow as tf
from skimage import io, filters
import numpy as np

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

def FilterImage(img = None, grayscale = True, Relu = True, Kmean=True, n_colors = 200):
    grayscale = rgb2gray(img)
    grayscale = filters.butterworth(grayscale)

    grayscale = np.array(grayscale, dtype=np.float64) / 255
    w, h = original_shape = tuple(grayscale.shape)
    image_array = np.reshape(grayscale, (w * h, 1))
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1000)
    kmeans = KMeans(n_clusters=5, n_init="auto", random_state=0).fit(
        image_array_sample
    )
    labels = kmeans.predict(image_array)
    new_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
    return new_image

def get_trainings_patches(complete_list):
    train_label_array = []
    patch_size = (32, 32)
    patches = []
    labels = []
    for image_number in range(len(complete_list[0])):
        image = complete_list[0][image_number]
        mask = complete_list[1][image_number]
        extra_layer = complete_list[2][image_number]
        image = FilterImage(image)
        extra_layer = np.array(extra_layer, dtype=np.float64)
        #image = np.array(image, dtype=np.float64) / 255
        for i in range(patch_size[0] + 1, image.shape[0], patch_size[0]):
            for j in range(patch_size[1] + 1, image.shape[1], patch_size[1]):
                patch = image[i - int(patch_size[0] / 2):i + int(patch_size[0] / 2),
                        j - int(patch_size[1] / 2):j + int(patch_size[1] / 2)]
                patch_extra_layer = extra_layer[i - int(patch_size[0] / 2):i + int(patch_size[0] / 2),
                        j - int(patch_size[1] / 2):j + int(patch_size[1] / 2)]
                if (patch.shape != (patch_size[0], patch_size[1], 1 )):
                    #print(patch.shape)
                    continue
                patch = np.reshape(patch, (32,32,1))
                patch = np.concatenate((patch, patch_extra_layer[:, :, np.newaxis]), axis=2)
                if (patch.shape != (patch_size[0], patch_size[1], 2)):
                    continue
                patches.append(patch)
                label = np.reshape(mask[i][j][0], 1)[0]
                labels.append(label)
    patches = np.array(patches)
    labels = np.array(labels)
    return (patches, labels)

