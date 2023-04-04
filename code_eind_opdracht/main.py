from data_inladen import get_list_of_mask_and_image , load_single_image, load_single_mask
from  make_patches import get_trainings_patches
from trainModel import  train_model
from make_segmentation_and_erode_mask import make_mask_list
from show_prediction import  show_and_make_predictions, make_mask_predictions, test_accuracy, make_mask_predictions_batch, test_accuracy_batch
import tensorflow as tf
from skimage.transform import resize
import numpy as np

tf.keras.backend.clear_session()
lijst = get_list_of_mask_and_image(0, 14)
lijst_test = get_list_of_mask_and_image(15, 20)
for i in range(0, len(lijst_test[0])):
    # test data kleiner maken want wordt per pixel voorspeld inplaats met stappen voor training
    # zonder dit duurt testen te lang
    lijst_test[0][i] = resize(lijst_test[0][i], (250, 250))
    lijst_test[1][i] = np.array(lijst_test[1][i], dtype=np.float64)
    lijst_test[1][i] = resize(lijst_test[1][i], (250, 250))

lijst.append(make_mask_list(lijst[0]))
for i in range(16, 33, 4):
    patch_size = (i,i)
    data_train = get_trainings_patches(lijst, patch_size= patch_size, extra_layer_bool=False)
    model = train_model(data_train[0], data_train[1], patch_size = patch_size,extra_layer_bool=False)
    masklist = make_mask_predictions_batch(lijst_test[0], patch_size=patch_size, extra_layer_bool=False, model=model)
    with open("output.txt", "a") as myfile:
        result = str(test_accuracy_batch(masklist, lijst_test[1], patch_size=patch_size) )  + " accuracy of prediction with patch " + str(patch_size) + "\n"
        myfile.write(result)
    tf.keras.backend.clear_session()

#patch_size = (8,8)
#image = load_single_image(25)
#true_mask = load_single_mask(25)
#mask = make_mask_predictions(image, patch_size= patch_size)
#test_accuracy(mask, true_mask, patch_size= patch_size)
#show_and_make_predictions(image, patch_size=patch_size)
# image = load_single_image(88)
# show_and_make_predictions(image)