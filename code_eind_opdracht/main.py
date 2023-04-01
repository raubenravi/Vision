from data_inladen import get_list_of_mask_and_image , load_single_image
from  make_patches import get_trainings_patches
from trainModel import  train_model
from make_segmentation_and_erode_mask import make_mask_list
from show_prediction import  show_and_make_predictions
# lijst = get_list_of_mask_and_image(0, 20)
# lijst.append(make_mask_list(lijst[0]))
# data_train = get_trainings_patches(lijst)
# train_model(data_train[0], data_train[1])


image = load_single_image(88)
show_and_make_predictions(image)