#%%
import os
import pandas as pd
import numpy as np

from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# import matplotlib.pyplot as plt 


#%%
mnist_path = os.getcwd() + "/fashion_mnist/data/fashion"

# labels 
labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

print("labels {}".format(labels))
#%%
# laod mnist as a tensorflow dataset 
data = input_data.read_data_sets(mnist_path)

# data contains ('train', 'validation', 'test')
print(
    "Train {}\n"
    "test {}\n" 
    "validation {}\n". 
    format( 
        data.train.num_examples,
        data.test.num_examples,
        data.validation.num_examples))

no_batches = 5
batch_size = 30

plot_cols = 5
plot_rows = 5 


for b in range(no_batches):
    batch_x = data.train.next_batch(batch_size)
    batch_x_images = batch_x[0]
    batch_x_labels = batch_x[1]
    #string for labels 
    batch_x_labels_str = np.array([labels[l] for l in batch_x_labels])

    # update number of cols 
    plot_rows = int(len(batch_x_images)/plot_cols) 

    fig, axs = plt.subplots(
        nrows=plot_rows, 
        ncols=plot_cols, 
        gridspec_kw={'wspace':0, 'hspace':1},
        squeeze=False)

    
    for axrow in range(plot_rows): 
        for axcol in range(plot_cols): 
            i = axrow*plot_cols+axcol

            # images in the tensorflow dataset are already scaled 
            img = (batch_x_images[i]).reshape(28,28) * 255
            axs[axrow, axcol].axis("off")
            axs[axrow, axcol].set_title(batch_x_labels_str[i])
            axs[axrow, axcol].imshow(img, cmap='gray', vmin=0, vmax=255)



# batch_1 = data.train.next_batch(10)
# batch_1_images = batch_1[0]
# batch_1_labels = batch_1[1]







