from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import mnist
import numpy as np
from typing import Tuple

def load_train(padding=((0, 0), (0, 0), (0, 3))) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns training data, X, y. 
    """
    #mnist.datasets_url = "https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data"
    train_images = mnist.train_images()
    mnist.train_labels()
    return np.pad(train_images, padding), mnist.load_data()

def load_test(padding=((0, 0), (0, 0), (3, 0))) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns testing data, X, y. 
    """
    mnist.datasets_url = "https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data"
    test_images = mnist.test_images()

    return np.pad(test_images, padding), mnist.test_labels()

def load_example(index=4, paddingL=((0,0), (0, 3)), paddingR=((0,0), (3, 0))):
    """
    Returns one image twice with different paddings
    """
    mnist.datasets_url = "https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data"
    example_image = mnist.test_images()[index]
    
    return (np.pad(example_image, paddingL), np.pad(example_image, paddingR)), mnist.test_labels()[index]