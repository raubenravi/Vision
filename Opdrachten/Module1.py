from skimage import data
import matplotlib.pyplot as plt
import skimage
import skimage.io as io

#image = io.imread('C:/Users/ruben/Downloads/len_full.jpg')
#imageOrg = io.imread('C:/Users/ruben/Downloads/len_full.jpg')
from skimage.color import rgb2hsv

image = data.astronaut()


def checkorange(red, green ,blue):
    """

    :param red: value of redpixel
    :param green: value of greenpixel
    :param blue: value of bluepixel

    :return: true if pixel has orange otherwihse false
    """
    #preventing dividing by zero
    red += 0.01
    green += 0.01
    blue += 0.01
    redPercentage = float(red) / (float(red) + float(green) + float(blue))
    greenPercentage = float(green) / (float(red) + float(green) + float(blue))
    bluePercentage = float(blue) / (float(red) + float(green) + float(blue))
    if (redPercentage >= 0.4 and greenPercentage >= 0.2 and bluePercentage <= 0.6):
        return True
    return 0
    if(redPercentage >= 0.4 and greenPercentage >= 0.2  and bluePercentage <= 0.6 ):
        return True
    return 0

def HiglightColor(image):
    """
    function to higlight the orange color in image

    :param image: color image
    :return: grayscale image except for the orange color images
    """

    # output: grayscale image execpt 1 collor
    for x in range(image.shape[0]):
        #ik loop door alle pixels heen en check als ze in een collor range zitten
        for y in range(image.shape[1]):
           red = image[x, y, 0]
           green = image[x, y, 1]
           blue =  image[x, y, 2]
           if (checkorange(red, green ,blue) == False):
               #standaard grayscale formule converter van documentatie
               greyscale = 0.2125 * red + 0.7154 * green + 0.0721 * blue
               image[x, y, 0] = greyscale
               image[x, y, 1] = greyscale
               image[x, y, 2] = greyscale
    return image



image = data.astronaut()
HiglightColor(image)
skimage.io.imshow(HiglightColor(image))
plt.show()
1#viewer.show()

hsv_img = rgb2hsv(image)
hue_img = hsv_img[:, :, -1]
viewer = skimage.io.imshow(hue_img)
plt.show()
#viewer.show()


fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))
ax0.hist(hue_img.ravel(), 512)
ax0.set_title("Histogram of the Hue channel with threshold")
ax0.set_xbound(0, 0.12)
fig.tight_layout()
plt.show()