from data_inladen import get_list_of_mask_and_image , load_single_image, load_single_mask
from  make_patches import get_trainings_patches
from trainModel import  train_model
from make_segmentation_and_erode_mask import make_mask_list
from show_prediction import  show_and_make_predictions, make_mask_predictions, test_accuracy, test_accuracy_eroded, test_accuracy_batch
import tensorflow as tf
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

plot_x_n_clusters = []
plot_y_accurcy = []

lijst_test = get_list_of_mask_and_image(15, 20)
for i in range(1, 15):
    lijst_mask = make_mask_list(lijst_test[0], n_clusters=i)
    accuracy = 0
    for j in range(0, len(lijst_test[0]) ):
        try:
            accuracy += test_accuracy_eroded(lijst_mask[j], lijst_test[1][j])
        except:
            pass
    accuracy = accuracy / len(lijst_test[0])
    print(accuracy, "accurcy of list with n of: ", i)
    plot_x_n_clusters.append(i)
    plot_y_accurcy.append(accuracy)


plt.plot(plot_x_n_clusters, plot_y_accurcy)
plt.show()