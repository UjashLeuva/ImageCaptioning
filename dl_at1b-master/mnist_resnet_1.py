#%%
from utils.load_mnist_1 import *

import numpy as np

#%% 
# import ResNet and configure it 
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam, Adadelta
from keras.applications import ResNet50
from keras.applications import resnet50 as resnet50
from keras.preprocessing import image
from keras.models import Model 
from keras.utils import to_categorical
from keras.regularizers import l2

#%%
print("loaded data set from mnist_1 method")

#%% function to load and prepare images for resnet_transfer
def resnet_transfer_preparedata(kind='train', no=0): 
    import skimage.transform

    imgs, lbls = [], []

    if kind == 'train': 
        imgs, lbls = load_images_train_32_32_rgb()
    elif kind == "test": 
        imgs, lbls = load_images_test_32_32_rgb()


    if no != 0: 
        imgs = imgs[:no]
        lbls = lbls[:no]

    imgs_resize = []
    no_images = len(imgs)
    status_print = "resizing {} / " +  str(no_images)
    print(status_print.format(0), end="\r")

    for i in range(len(imgs)): 
        img = skimage.transform.resize(
            imgs[i], 
            (38,38),
            mode ='constant')
        imgs_resize.append(img)
        if i % 500 == 0: 
            print(status_print.format(i), end="\r")

    imgs_resize = np.array(imgs_resize)
    print("resizing complete: " + str(imgs_resize.shape))

    lbls_str = image_class_to_str(lbls)
    return imgs_resize, lbls, lbls_str

#%%
train_images, train_labels, train_labels_str = resnet_transfer_preparedata(no=1000)
test_images, test_labels, test_labels_str = resnet_transfer_preparedata('test', no=250)

train_labels_cat = to_categorical(train_labels, num_classes=10)
test_labels_cat = to_categorical(test_labels, num_classes=10)

#%% prepare new model 


base_model = ResNet50(
    include_top = False, 
    weights = 'imagenet', 
    input_shape = [38,38,3]
)

#%% freeze: set base_model layers to non trainable 
for l in base_model.layers[:]: 
    l.trainable = False

#%%
base_model.summary()

#%%

added_layers = base_model.output 

# Prior to GAP, one would flatten your tensor and then add a few fully connected layers in your model. 
# The problem is that a bunch of parameters in your model end up being attributed to the dense layers and 
# could potentially lead to overfitting. A natural solution was to add dropout to help regulate that.

# However a few years ago, the idea of having a Global Average Pooling came into play. 
# GAP can be viewed as alternative to the whole flatten FC Dropout paradigm. GAP helps prevent overfitting 
# by doing an extreme form of reduction. Given a H X W X D tensor, GAP will average the H X W features 
# into a single number and reduce the tensor into a 1 X 1 X D tensor.

# The original paper simply applied GAP and then a softmax. However, it's now common to have GAP followed by a FC layer.


added_layers = GlobalAveragePooling2D()(added_layers)#  Flatten()(added_layers)
added_layers = Dropout(0.7)(added_layers)

# initial run was showing very high variance (acc on trainng is good, but acc on val is very poor)
# so i added regulisation 
added_layers = Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(added_layers)
added_layers = Activation('relu')(added_layers)
# added_layers = BatchNormalization()(added_layers)

preds = Dense(10, activation ='softmax')(added_layers)

final_model = Model(input = base_model.input, outputs=preds)

final_model.summary()

#%% compile the model 
adam = Adam(lr=0.0001)
final_model.compile(optimizer= Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

#%% 
train_images = train_images[:1000]
train_labels_cat = train_labels_cat[:1000]

test_images = test_images[:250]
test_labels_cat = test_labels_cat[:250]

print(train_images.shape)
print(test_images.shape)
print(train_labels_cat.shape)
print(test_labels_cat.shape)

print(np.max(train_images), np.min(train_images))
print(np.max(test_images), np.min(test_images))
#%% fit the model 
history = final_model.fit(
    train_images, train_labels_cat, 
    validation_data= (test_images, test_labels_cat ), 
    epochs = 15, batch_size = 100)

#%%
# We can get our score
score = final_model.evaluate(test_images, test_labels_cat, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
# https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#%% test predict

pred_output = final_model.predict(train_images[:1])



#%%
