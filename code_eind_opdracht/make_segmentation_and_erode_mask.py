from skimage.morphology import square
import skimage.morphology
import numpy as np
import skimage.io
from skimage.color import rgb2hsv, rgb2gray
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)


def splitIntoReducedColors(img, kmean_clusters = 20):

    img = np.array(img, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(img.shape)
    image_array = np.reshape(img, (w * h, -1))
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=kmean_clusters, n_init="auto", random_state=0).fit(
        image_array_sample)

    labels = kmeans.predict(image_array)
    new_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
    #print(kmeans.cluster_centers_.shape)
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
    #skimage.io.imshow(best_Mask)
    #plt.show()
    return best_Mask


def make_mask_list(image_list, n_clusters=20):
    list_with_best_masks = []
    for i in range(len(image_list)):
        masks = splitIntoReducedColors(image_list[i], kmean_clusters=n_clusters)
        list_with_best_masks.append(EvulateMask(masks=masks))
    return list_with_best_masks

def make_mask(image, n_clusters=20):
    return EvulateMask(masks = splitIntoReducedColors(image, kmean_clusters=n_clusters))