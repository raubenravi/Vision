from skimage import data, filters, feature
from skimage.viewer import ImageViewer
import scipy
from scipy import ndimage
from skimage.color import rgb2gray
import  skimage.io as io




def apllyMaskAndShow(img, mask):
    """
    function to use mask on filter and show it on the screen

    @:param image, iamge color image to apply filter on
    @:param mask  , mask to apply on images
    
    :return: grayscale image convolved with the filter
    """
    img = rgb2gray(img)
    img = scipy.ndimage.convolve(img, mask)
    viewer = ImageViewer(img)
    viewer.show()
    return img


#opdracht 1, first just the standard lubrary filter
image = data.camera()
image = filters.sobel(image)
viewer = ImageViewer(image)
viewer.show()

#opdracht 2
#image = io.imread('C:/Users/ruben/Downloads/len_full.jpg')
#some random mask from wikipedia
GausianFilterMask=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
mask2=[[1,1,1],[1,-8,1],[1,1,1]]
mask3=[[0,-1,0],[1,-4,1],[0,-1,0]]
mask4=[[1,0,-1],[0,0,0],[-1,0,1]]

#Let`s test those masks
apllyMaskAndShow(image, GausianFilterMask)
apllyMaskAndShow(image, mask2)
apllyMaskAndShow(image, mask3)
apllyMaskAndShow(image, mask4)

#opdracht 3

image = feature.canny(image, sigma=1.9)
viewer = ImageViewer(image)
viewer.show()

