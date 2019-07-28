#%%
from utils.load_mnist_1 import *

#%%

# https://betweenandbetwixt.com/2019/01/04/convolutional-neural-network-with-keras-mnist/
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam, SGD, Adadelta


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPool2D, BatchNormalization
from keras.models import Sequential

from keras.utils import to_categorical

#%% import regulisatiers
from keras.regularizers import l2



#%%
print("loaded data set from mnist_1 method")
print("introduce regulisation")
# https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/

#%%
train_images, train_labels = load_images_train()
train_labels_str = image_class_to_str(train_labels)

test_images, test_labels = load_images_test()
test_lables_str = image_class_to_str(test_labels)


#%% 
# reshape inputs 
no_images_train = len(train_images)
no_images_test = len(test_images)


train_images_reshape = train_images.reshape(no_images_train, 28,28,1)
test_images_reshape = test_images.reshape(no_images_test, 28,28,1)



train_labels_cat = to_categorical(train_labels, num_classes=10)
test_labels_cat = to_categorical(test_labels, num_classes=10)
print(train_labels[10])
print(train_labels_cat[10])

print(train_labels[5])
print(train_labels_cat[5])

train_images_reshape = train_images_reshape / 255
test_images_reshape = test_images_reshape / 255

#%%

# keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

model = Sequential()
model.add(
    Conv2D(
        filters=8, 
        kernel_size=3, 
        padding="same", 
        input_shape=(28,28,1), 
        kernel_initializer='he_normal',
        data_format="channels_last"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(filters=16, kernel_size=3, padding="same",  kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(filters=16, kernel_size=3, padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation("relu"))


model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Flatten()) # Stretching out for our FC layer
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation("relu"))
		
model.add(Dense(10,activation='softmax'))




# print the cnn arch
model.summary()

model.compile(loss="categorical_crossentropy",
            optimizer=Adadelta(),
            metrics=["accuracy"])
# Test accuracy: 0.9157 5 epoch
# Test accuracy: 0.918 10 epoch


#%%
history = model.fit(
    train_images_reshape, 
    train_labels_cat, 
    epochs=20,
    batch_size=128,
    validation_data=(test_images_reshape, test_labels_cat))

#%%
# We can get our score
score = model.evaluate(test_images_reshape, test_labels_cat, verbose=0)

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

#%%
