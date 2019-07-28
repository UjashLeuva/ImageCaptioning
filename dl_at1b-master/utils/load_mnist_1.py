#%%
import os
import pandas as pd
import numpy as np
from fashion_mnist.utils.mnist_reader import load_mnist

from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


#%%
# load labels text 
def image_class_to_str(image_class_set, labels=['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']):
    image_class_str = np.array([labels[j] for j in image_class_set])
    return(image_class_str)

#
#%%
# plot a small batch to verfiy that labels match the images  
# and we loaded things correctly 
#images_img=train_images, labels_str=train_labels_str
def plot_images(images_img, labels_str, imgs = 20, cols=5):

    no_imgs = imgs
    plot_cols = cols
    plot_rows = int(no_imgs / plot_cols)

    fig, axs = plt.subplots(
        nrows=plot_rows, 
        ncols=plot_cols, 
        gridspec_kw={'wspace':0, 'hspace':1},
        squeeze=False)

    for axrow in range(plot_rows): 
        for axcol in range(plot_cols): 
            i = axrow*plot_cols+axcol
            # print("{} - image class {}".format(i, train_labels_str[i]))
            #print(axrow, axcol, axrow*width+axcol)
            img = (images_img[i]).reshape(28,28)
            axs[axrow, axcol].axis("off")
            axs[axrow, axcol].set_title(labels_str[i])
            axs[axrow, axcol].imshow(img, cmap='gray', vmin=0, vmax=255)

    
    plt.show()  



#%%
def load_images_train():
    # returns images, label_classes
    return load_images()

def load_images_test():
    return load_images(kind="t10k")


def load_images_train_32_32_rgb():
    # returns images, label_classes
    return load_images_32_32_rgb()

def load_images_test_32_32_rgb():
    return load_images_32_32_rgb(kind="t10k")



def load_images(path= "/fashion_mnist/data/fashion", kind='train'): 

    mnist_path = os.getcwd() + path
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    # reading the data using the utility function load_mnist
    # train_images, train_labels  = load_mnist(mnist_path, kind='train')
    # t10k_images, t10k_labels  = load_mnist(mnist_path, kind='t10k')
    # train_labels_str = image_class_to_str(train_labels)
    # t10k_labels_str = image_class_to_str(t10k_labels)

    return load_mnist(mnist_path, kind=kind)

def load_images_32_32_rgb(path= "/fashion_mnist/data/fashion", kind='train'):
    imgs, lbls = load_images(path=path, kind=kind)
    imgs = imgs.reshape(len(imgs), 28,28)
    imgs = np.stack((imgs,)*3, axis = -1)
    imgs = np.pad(
        imgs, 
        pad_width = ((0,0),(2,2),(2,2),(0,0)),
        mode ='constant',
        constant_values = 0)
    return imgs, lbls

#%% 
def main(): 
    print("main load mnist_1")
    
    train_images, train_labels = load_images_train()
    train_labels_str = image_class_to_str(train_labels)

    test_images, test_labels = load_images_test()
    test_lables_str = image_class_to_str(test_labels)



    print("plot train images")
    plot_images(train_images, train_labels_str)
    print("plot test images - more")
    plot_images(test_images, test_labels, imgs=10, cols=5)



#%% 
if __name__ == '__main__': 
    main() 