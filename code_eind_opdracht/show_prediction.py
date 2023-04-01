import numpy as np
import  tensorflow as tf
from make_segmentation_and_erode_mask import make_mask
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from make_patches import FilterImage

def make_patches_prediction(image):
    extra_layer = make_mask(image)
    image= FilterImage(image)
    #image = rgb2gray(image)
    #image = np.array(image, dtype=np.float64) / 255
    extra_layer = np.array(extra_layer, dtype=np.float64)
    patches = []
    patch_size = (32,32)
    for i in range(patch_size[0] + 1 , image.shape[0] - patch_size[0]):
        for j in range(patch_size[1] + 1, image.shape[1] - patch_size[1]):
            patch = image[i - int(patch_size[0] / 2):i + int(patch_size[0] / 2),
                    j - int(patch_size[1] / 2):j + int(patch_size[1] / 2)]
            patch_extra_layer = extra_layer[i - int(patch_size[0] / 2):i + int(patch_size[0] / 2),
                                j - int(patch_size[1] / 2):j + int(patch_size[1] / 2)]
            #print(patch.shape)
            #if (patch.shape != (patch_size[0], patch_size[1])):
            if (patch.shape != (patch_size[0], patch_size[1], 1)):
                print(patch.shape)
                patch = np.zeros((int(patch_size[0]), int(patch_size[1])))
                patch_extra_layer = np.zeros((int(patch_size[0]), int(patch_size[1])))
                print(i , " ", j)
                #continue
            patch = np.reshape(patch, (32, 32, 1))
            patch = np.concatenate((patch, patch_extra_layer[:, :, np.newaxis]), axis=2)
            if (patch.shape != (patch_size[0], patch_size[1], 2)):
                print("fout")
                continue
            patches.append(patch)
    print(len(patches))
    patches = np.array(patches)
    return patches

def get_model(path='saved_model/my_model'):
    return tf.keras.models.load_model(path)

def make_predictions(patches, original_image ,model = get_model()):
    original_image = rgb2gray(original_image)
    predictions = model.predict(patches)
    return np.reshape(predictions, tuple(int(i) for i in (original_image.shape[0] - 65, original_image.shape[1] - 65, -1)))

def show_and_make_predictions(image):
    image = resize(image, (500,500))
    original = image
    patches = make_patches_prediction(image)
    preds = make_predictions(patches, original)
    #threshold = 140
    #preds = np.where(preds > threshold, 1, 0)
    #preds = np.uint8(255 * preds[:, :])
    #preds = np.mean(preds, axis=2)
    #preds = np.uint8(255 * preds[:, :])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].imshow(original[:, :, :])
    ax[0].set_title("Original")
    ax[1].set_title("prediction")
    ax[1].imshow(preds[:,:])
    plt.show()