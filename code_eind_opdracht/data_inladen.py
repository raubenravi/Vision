import pandas as pd
import skimage.io as io
import tensorflow as tf
from skimage.color import rgb2gray
import numpy as np
def load_image(image_name, path):
    image = io.imread(path + image_name)[:,:,:3]
    return image

def load_mask(mask_name, path):
    mask = tf.io.read_file(path + mask_name)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, 1, 0)

    return mask

#zo staat het in mijn folder
#\Vision\code_eind_opdracht\data\Mask\0.png"
def get_list_of_mask_and_image(start, end, path="./data"):
    path_meta_data = path + "/metadata.csv"
    df_meta_data = pd.read_csv(path_meta_data)
    df_meta_data = df_meta_data.drop([28])
    df_meta_data = df_meta_data.drop([22])
    df_meta_data = df_meta_data.drop([19])
    df_meta_data = df_meta_data.drop([14])
    df_meta_data = df_meta_data.drop([9])
    selection = df_meta_data[start:end + 1]
    image_list = []
    mask_list = []
    #name_list = []
    for row in selection.iterrows():
        try:
            image = load_image(row[1]['Image'], path + "/Image/")
            mask = load_mask(row[1]['Mask'], path + "/Mask/")
        except:
            print("error with ", row[1])
        image_list.append(image)
        mask_list.append(mask)
        #str = (row[1]['Image'] + row[1]['Mask'])
        #name_list.append(str)
        image = None
        mask = None
    list_all = [image_list, mask_list]
    return list_all

def load_single_image(image_number,  path="./data"):
    path_meta_data = path + "/metadata.csv"
    df_meta_data = pd.read_csv(path_meta_data)
    row = df_meta_data.loc[image_number]
    return load_image(row['Image'], path + "/Image/")

def load_single_mask(mask_number,  path="./data"):
    path_meta_data = path + "/metadata.csv"
    df_meta_data = pd.read_csv(path_meta_data)
    row = df_meta_data.loc[mask_number]
    return load_mask(row['Mask'], path + "/Mask/")
