import numpy as np
import  tensorflow as tf
from make_segmentation_and_erode_mask import make_mask
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from make_patches import FilterImage

def make_patches_prediction(image, patch_size, extra_layer_bool=True):
    extra_layer = make_mask(image, n_clusters=6 )
    w, h, d = original_shape = tuple(image.shape)
    image = rgb2gray(image)
    image = np.array(image, dtype=np.float64)
    image = np.reshape(image, (w, h, 1))
    extra_layer = np.array(extra_layer, dtype=np.float64)
    patches = []
    for i in range(patch_size[0] + 1 , image.shape[0] - patch_size[0]):
        for j in range(patch_size[1] + 1, image.shape[1] - patch_size[1]):
            patch = image[i - int(patch_size[0] / 2):i + int(patch_size[0] / 2),
                    j - int(patch_size[1] / 2):j + int(patch_size[1] / 2)]
            if (extra_layer_bool):
                patch_extra_layer = extra_layer[i - int(patch_size[0] / 2):i + int(patch_size[0] / 2),
                                    j - int(patch_size[1] / 2):j + int(patch_size[1] / 2)]
            #print(patch.shape)
            #if (patch.shape != (patch_size[0], patch_size[1])):
            if (patch.shape != (patch_size[0], patch_size[1], 1)):
                patch = np.zeros((int(patch_size[0]), int(patch_size[1])))
                patch_extra_layer = np.zeros((int(patch_size[0]), int(patch_size[1])))
                print(i , " ", j)
            patch = np.reshape(patch, (patch_size[0], patch_size[1], 1))
            if(extra_layer_bool):
                patch = np.concatenate((patch, patch_extra_layer[:, :, np.newaxis]), axis=2)
                if (patch.shape != (patch_size[0], patch_size[1], 2)):
                    print("fout")
                    continue
            else:
                if (patch.shape != (patch_size[0], patch_size[1], 1)):
                    continue
            patches.append(patch)
    #print(len(patches))
    patches = np.array(patches)
    return patches

def get_model(path='saved_model/my_model'):
    try:
        return tf.keras.models.load_model(path)
    except:
        print("error loading model")

def make_predictions(patches, original_image , patch_size ,model=get_model()):
    original_image = rgb2gray(original_image)
    predictions = model.predict(patches)
    return np.reshape(predictions, tuple(int(i) for i in (original_image.shape[0] - (patch_size[0] * 2 +1), original_image.shape[1] - (patch_size[1] * 2 +1), -1)))

def make_mask_predictions(image, patch_size = (32,32), extra_layer_bool=True , model=get_model() ):
    original = image
    patches = make_patches_prediction(image, patch_size, extra_layer_bool)
    return make_predictions(patches, original, patch_size, model=model)

def make_mask_predictions_batch(image_list, patch_size = (32,32), extra_layer_bool=True, model =get_model()):
    result_list = []
    #print(len(image_list))
    for image in image_list:
        original = image
        patches = make_patches_prediction(image, patch_size, extra_layer_bool)
        result_list.append(  make_predictions(patches, original, patch_size, model=model ))
    return result_list

def show_and_make_predictions(image, patch_size = (32,32),  extra_layer_bool=True):
    #image = resize(image, (500,500))
    original = image
    patches = make_patches_prediction(image , patch_size, extra_layer_bool= extra_layer_bool)
    preds = make_predictions(patches, original, patch_size)
    mask = make_mask_predictions(image, patch_size=patch_size, extra_layer_bool=False)
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].imshow(original[:, :, :])
    ax[0].set_title("Original")
    ax[1].set_title("prediction 1")
    ax[1].imshow(preds[:,:], cmap='gray')
    ax[2].imshow(mask[:, :], cmap='gray')
    ax[2].set_title("prediction 2")
    ax[2].imshow(preds[:, :], cmap='gray')
    plt.show()

def test_accuracy_batch(mask_list, true_mask_list, threshold = 0.5, patch_size=(32, 32)):
    accuracy = 0
    for i in range(0, len(mask_list) ):
        accuracy += test_accuracy(mask_list[i], true_mask_list[i], threshold=threshold, patch_size=patch_size)
    return accuracy / len(mask_list)

def test_accuracy(mask, true_mask, threshold = 0.5, patch_size=(32,32) ):
    true_mask = true_mask[patch_size[0] + 1:-patch_size[0], patch_size[1] + 1:-patch_size[1], :]
    true_mask = np.array(true_mask, dtype=np.float32)
    mask = np.where(mask > threshold, 1, 0)
    mask_inverted = np.where(mask > threshold, 0, 1)
    accuracy_mask = mask * true_mask
    true_positif = (np.count_nonzero(accuracy_mask))
    true_mask_inverted = np.where(true_mask == 0, 1, 0)
    accuracy_mask = mask_inverted * true_mask_inverted
    true_negative = (np.count_nonzero(accuracy_mask))
    return ( (true_positif + true_negative) / true_mask.size)


def test_accuracy_eroded(mask, true_mask, threshold = 0.5 ):
    true_mask = np.array(true_mask, dtype=np.float32)
    mask = np.array(mask)
    #print(mask.shape)
    mask = np.reshape(mask, (mask.shape[0],mask.shape[1],1) )
    mask = np.where(mask > threshold, 1, 0)
    mask_inverted = np.where(mask > threshold, 0, 1)
    accuracy_mask = mask * true_mask
    true_positif = (np.count_nonzero(accuracy_mask))
    true_mask_inverted = np.where(true_mask == 0, 1, 0)
    accuracy_mask = mask_inverted * true_mask_inverted
    true_negative = (np.count_nonzero(accuracy_mask))
    return ( (true_positif + true_negative) / true_mask.size)