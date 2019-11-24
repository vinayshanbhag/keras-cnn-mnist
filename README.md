
## Introduction

MNIST is the "hello world" of image classification. It is a dataset of handwritten digits taken mostly from United States Census Bureau employees. It is a set of images of digits 0-9, in grayscale and exactly identical dimensions, 28pixels x 28pixels. This serves as a relatively simple test for machine learning experiments and evaluating different models.

In this notebook, I explore convolutional neural networks using Keras. Specifically, learning rate annealing and data augmentation. While I expect to see fairly high accuracy, my primary objective here is to figure out the impact of these techniques on classification outcomes.


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
```


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import numpy as np
from numpy import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, Callback
from keras import regularizers
from keras.optimizers import Adam


## visualize model using GraphViz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model

def display_images(X, y=[], rows=5, columns=5, cmap="gray"):
    """ Display images and labels
    """
    fig, ax = plt.subplots(rows,columns, figsize=(6,6))
    for row in range(rows):
        for column in range(columns):
            ax[row][column].imshow(X[(row*columns)+column].reshape(28,28), cmap=cmap)
            ax[row][column].set_axis_off()
            if len(y):ax[row][column].set_title("{}:{}".format("label",np.argmax(y[(row*columns)+column])))
    fig.tight_layout()

%matplotlib inline
```

## Load, prepare and preview data

Our data looks like the sample below. It is a csv file with the true classification in the ```label``` column, followed by 784 pixel values (28x28 pixels unrolled). Each pixel takes a value ranging from 0-255. Since these are black and white images, each pixel is represented by a single value (channel) instead of three separate R, G, B values (3 channels) in a color image.


```python
#df = pd.read_csv("../input/train.csv")
df = pd.read_csv("train.csv")
df.sample(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22820</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 785 columns</p>
</div>



We pick random samples from these 42000 images to create 3 sets - 
- training (60%): data used to train convnet
- cross validation (20%): data used to validate performance
- test (20%): data used to test classification accuracy

While there is a separate test set available, we are not using that in this notebook, since it is not labeled and cannot be easily evaluated.



```python
X_train, X_val, y_train, y_val = train_test_split(df.iloc[:,1:].values, df.iloc[:,0].values, test_size = 0.4)
X_cv, X_test, y_cv, y_test = train_test_split(X_val, y_val, test_size = 0.5)
print("X_train:{}\ny_train:{}\n\nX_cv:{}\ny_cv:{}\n\nX_test:{}\ny_test:{}".format(X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape))
```

    X_train:(25200, 784)
    y_train:(25200,)
    
    X_cv:(8400, 784)
    y_cv:(8400,)
    
    X_test:(8400, 784)
    y_test:(8400,)
    

The data is in an unrolled format, i.e. each sample is a sequence of 784 pixel values. We will convert this using numpy's reshape function to (28x28x1). i.e. an image that is 28 pixels wide and 28 pixels tall, with 1 channel (black and white image).  So for example, the shape of the training set becomes (25200 samples, 28px, 28px, 1ch)

We change the output class (label) to categorical or one hot format. i.e. instead of a single value 0-9, we convert this to a array of size 10. e.g.
y = [9] becomes
y = [0,0,0,0,0,0,0,0,1,0]

Additionally, we scale all the features (pixel values) from a range of 0-255, to a range of 0-1. This is done by dividing each value in the feature matrix by 255.

Here are the new shapes of training, cross validation and test data sets.


```python
width = 28
height = 28
channels = 1
X_train = X_train.reshape(X_train.shape[0], width, height, channels)
X_cv = X_cv.reshape(X_cv.shape[0], width, height, channels)
X_test = X_test.reshape(X_test.shape[0], width, height, channels)

# convert output classes to one hot representation
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_cv = np_utils.to_categorical(y_cv, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

X_train = X_train.astype('float32')
X_cv = X_cv.astype('float32')
X_test = X_test.astype('float32')

# Scale features (pixel values) from 0-255, to 0-1 
X_train /= 255
X_cv /= 255
X_test /= 255
print("Reshaped:")
print("X_train:{}\ny_train:{}\n\nX_cv:{}\ny_cv:{}\n\nX_test:{}\ny_test:{}".format(X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape))
```

    Reshaped:
    X_train:(25200, 28, 28, 1)
    y_train:(25200, 10)
    
    X_cv:(8400, 28, 28, 1)
    y_cv:(8400, 10)
    
    X_test:(8400, 28, 28, 1)
    y_test:(8400, 10)
    

Here is a preview of a few images in the training set.


```python
display_images(X_train, y_train)
```


![png](output_10_0.png)



```python
batch_size=64
epochs=20
verbose=2

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def create_model():
    m = Sequential()
    m.add(Conv2D(32, (5,5), padding="same", activation='relu', input_shape=(width, height, channels) ))
    m.add(Conv2D(32, (5,5), padding="same", activation='relu'))
    m.add(MaxPooling2D(pool_size=(2,2)))
    m.add(Dropout(0.25))
    m.add(Conv2D(64, (3,3), padding="same", activation='relu'))
    m.add(Conv2D(64, (3,3), padding="same", activation='relu'))
    m.add(MaxPooling2D(pool_size=(2,2)))
    m.add(Dropout(0.3))
    m.add(Flatten())
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(10, activation='softmax'))
    
    opt = "adam" #Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    m.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return m

def plot_metrics(h, title=""):
    """ Plot training metrics - loss and accuracy, for each epoch, 
        given a training history object
    """
    fig, axes = plt.subplots(1,2, figsize=(10,5))
      
    axes[0].plot(h.history['loss'], color="steelblue", label="Training", lw=2.0)
    axes[0].plot(h.history['val_loss'], color="orchid", label="Validation", lw=2.0)

    axes[0].set_title("{} (Loss)".format(title))
    axes[0].set_xlabel("Epoch")
    axes[0].set_xticks(np.arange(len(h.history["loss"]), 2))
    axes[0].set_ylabel("Loss")
    
    axes[1].plot(h.history['acc'], color="steelblue", label="Training", lw=2.0)
    axes[1].plot(h.history['val_acc'], color="orchid", label="Validation", lw=2.0)
    
    axes[1].set_title("{} (Accuracy)".format(title))
    axes[1].set_xlabel("Epoch")
    axes[1].set_xticks(np.arange(len(h.history["acc"]), 2))
    axes[1].set_ylabel("Accuracy")
    

    for axis in axes:
        axis.ticklabel_format(useOffset=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc='best', shadow=False)
    fig.tight_layout()
    
def plot_losses(batch_hist):
    fig, ax1 = plt.subplots()

    ax1.semilogx(batch_hist.losses)
    ax1.set_title("Loss history")  
    
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.show()
```

## Model Definition

I use this model as the starting point, built using Keras sequential API. I create 3 separate instances of this model and then compare results with learning rate annealing and image data augmentation. 

Keras also provides a easy way to generate this diagram from the model. See code below.

<img src="https://raw.githubusercontent.com/vinayshanbhag/keras-cnn-mnist/master/dm.png"/>


```python
dm = create_model()
#dm.summary()

dm_batch_hist = LossHistory()

metrics_deep = dm.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_cv, y_cv), verbose = verbose, shuffle=True, callbacks=[dm_batch_hist])
```

    Train on 25200 samples, validate on 8400 samples
    Epoch 1/20
    5s - loss: 0.3061 - acc: 0.9035 - val_loss: 0.0652 - val_acc: 0.9798
    Epoch 2/20
    5s - loss: 0.0921 - acc: 0.9719 - val_loss: 0.0494 - val_acc: 0.9855
    Epoch 3/20
    5s - loss: 0.0692 - acc: 0.9782 - val_loss: 0.0565 - val_acc: 0.9832
    Epoch 4/20
    5s - loss: 0.0575 - acc: 0.9818 - val_loss: 0.0593 - val_acc: 0.9833
    Epoch 5/20
    5s - loss: 0.0494 - acc: 0.9851 - val_loss: 0.0439 - val_acc: 0.9868
    Epoch 6/20
    5s - loss: 0.0417 - acc: 0.9872 - val_loss: 0.0334 - val_acc: 0.9905
    Epoch 7/20
    4s - loss: 0.0384 - acc: 0.9888 - val_loss: 0.0429 - val_acc: 0.9874
    Epoch 8/20
    5s - loss: 0.0358 - acc: 0.9890 - val_loss: 0.0483 - val_acc: 0.9873
    Epoch 9/20
    5s - loss: 0.0306 - acc: 0.9905 - val_loss: 0.0471 - val_acc: 0.9894
    Epoch 10/20
    5s - loss: 0.0299 - acc: 0.9901 - val_loss: 0.0329 - val_acc: 0.9899
    Epoch 11/20
    4s - loss: 0.0244 - acc: 0.9926 - val_loss: 0.0451 - val_acc: 0.9900
    Epoch 12/20
    5s - loss: 0.0245 - acc: 0.9929 - val_loss: 0.0388 - val_acc: 0.9898
    Epoch 13/20
    5s - loss: 0.0225 - acc: 0.9929 - val_loss: 0.0411 - val_acc: 0.9911
    Epoch 14/20
    5s - loss: 0.0238 - acc: 0.9924 - val_loss: 0.0343 - val_acc: 0.9915
    Epoch 15/20
    4s - loss: 0.0225 - acc: 0.9929 - val_loss: 0.0447 - val_acc: 0.9902
    Epoch 16/20
    5s - loss: 0.0205 - acc: 0.9939 - val_loss: 0.0387 - val_acc: 0.9913
    Epoch 17/20
    5s - loss: 0.0192 - acc: 0.9943 - val_loss: 0.0401 - val_acc: 0.9898
    Epoch 18/20
    5s - loss: 0.0190 - acc: 0.9938 - val_loss: 0.0410 - val_acc: 0.9911
    Epoch 19/20
    5s - loss: 0.0169 - acc: 0.9951 - val_loss: 0.0415 - val_acc: 0.9910
    Epoch 20/20
    5s - loss: 0.0159 - acc: 0.9959 - val_loss: 0.0450 - val_acc: 0.9896
    


```python
plot_losses(dm_batch_hist)
```


![png](output_14_0.png)

# Visualize model using GraphViz
plot_model(dm, show_shapes=True, show_layer_names=False,to_file='dm.png')
## Learning Rate Annealing

Learning rate is the step size in gradient descent. If the step size is too large, the system may oscillate chaotically. On the other hand, if the step size is too small, it may take too long or may settle on a local minimum. 

<p style="float: left;">We will watch validation accuracy in each epoch, and reduce the learning rate to a third, if it plateaus in 2 consecutive epochs. Keras provides an aptly named, ```ReduceLROnPlateau```, callback to adapt the learning rate based on results from each epoch. Ref: [```ReduceLROnPlateau```](https://keras.io/callbacks/#reducelronplateau) for more options.</p>

<p style="float: left">The verbose mode, allows us to see when this actually kicks in.</p>


```python
lrc = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=verbose, factor=0.5, min_lr=0.00001, epsilon=0.001)
```

## Model with learning rate annealing

We create a new instance of the same model, but this time, insert a callback to our learning rate control function defined above. Then fit the model to our training data set and collect metrics.


```python
dmlrc = create_model()
dmlrc_batch_hist = LossHistory()
metrics_deep_lrc = dmlrc.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_cv, y_cv), verbose = verbose, shuffle=True, callbacks=[lrc,dmlrc_batch_hist])
#dmlrc.save_weights("deep_lrc.h5")
```

    Train on 25200 samples, validate on 8400 samples
    Epoch 1/20
    5s - loss: 0.3119 - acc: 0.8998 - val_loss: 0.0777 - val_acc: 0.9758
    Epoch 2/20
    5s - loss: 0.0935 - acc: 0.9717 - val_loss: 0.0584 - val_acc: 0.9796
    Epoch 3/20
    5s - loss: 0.0708 - acc: 0.9796 - val_loss: 0.0502 - val_acc: 0.9845
    Epoch 4/20
    4s - loss: 0.0581 - acc: 0.9819 - val_loss: 0.0419 - val_acc: 0.9869
    Epoch 5/20
    4s - loss: 0.0474 - acc: 0.9848 - val_loss: 0.0405 - val_acc: 0.9885
    Epoch 6/20
    4s - loss: 0.0393 - acc: 0.9877 - val_loss: 0.0408 - val_acc: 0.9875
    Epoch 7/20
    5s - loss: 0.0370 - acc: 0.9886 - val_loss: 0.0431 - val_acc: 0.9864
    Epoch 8/20
    5s - loss: 0.0321 - acc: 0.9892 - val_loss: 0.0358 - val_acc: 0.9890
    Epoch 9/20
    5s - loss: 0.0303 - acc: 0.9908 - val_loss: 0.0391 - val_acc: 0.9895
    Epoch 10/20
    4s - loss: 0.0255 - acc: 0.9921 - val_loss: 0.0359 - val_acc: 0.9908
    Epoch 11/20
    
    Epoch 00010: reducing learning rate to 0.0005000000237487257.
    5s - loss: 0.0291 - acc: 0.9907 - val_loss: 0.0361 - val_acc: 0.9906
    Epoch 12/20
    4s - loss: 0.0165 - acc: 0.9944 - val_loss: 0.0402 - val_acc: 0.9896
    Epoch 13/20
    
    Epoch 00012: reducing learning rate to 0.0002500000118743628.
    4s - loss: 0.0113 - acc: 0.9965 - val_loss: 0.0408 - val_acc: 0.9912
    Epoch 14/20
    5s - loss: 0.0090 - acc: 0.9971 - val_loss: 0.0390 - val_acc: 0.9921
    Epoch 15/20
    
    Epoch 00014: reducing learning rate to 0.0001250000059371814.
    5s - loss: 0.0077 - acc: 0.9976 - val_loss: 0.0391 - val_acc: 0.9912
    Epoch 16/20
    4s - loss: 0.0064 - acc: 0.9977 - val_loss: 0.0388 - val_acc: 0.9921
    Epoch 17/20
    
    Epoch 00016: reducing learning rate to 6.25000029685907e-05.
    5s - loss: 0.0067 - acc: 0.9981 - val_loss: 0.0366 - val_acc: 0.9915
    Epoch 18/20
    4s - loss: 0.0059 - acc: 0.9981 - val_loss: 0.0378 - val_acc: 0.9923
    Epoch 19/20
    
    Epoch 00018: reducing learning rate to 3.125000148429535e-05.
    5s - loss: 0.0055 - acc: 0.9985 - val_loss: 0.0370 - val_acc: 0.9918
    Epoch 20/20
    4s - loss: 0.0045 - acc: 0.9987 - val_loss: 0.0390 - val_acc: 0.9918
    


```python
plot_losses(dmlrc_batch_hist)
```


![png](output_20_0.png)


## Data Augmentation
To improve classification accuracy, we can augment the training samples, with random transformations of images in the training set. In Keras, this is done using ```keras.preprocessing.image.ImageDataGenerator``` class. We can apply random transformations such as, zooming, rotation, shifting the image up/down. We will limit rotation to a few degrees, and disable horizontal and vertical flipping, as our dataset of digits is prone to produce ambiguous results with these operations. 

See [```ImageDataGenerator```](https://keras.io/preprocessing/image/#imagedatagenerator) for lots of other options that are useful for other types of images.


```python
idg = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.05, 
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        vertical_flip=False, data_format="channels_last")
```

Here are a few images produced by the image data generator.


```python
image_data = idg.flow(X_train,y_train, batch_size=25).next()
print("Sample images from ImageDataGenerator:")
display_images(image_data[0], image_data[1])
```

    Sample images from ImageDataGenerator:
    


![png](output_24_1.png)


## Model with data augmentation

We create yet another instance of the model we defined earlier with learning rate annealer. This time instead of fitting it to the training data set, we will instead fit it to the images generated by the ```ImageDataGenerator```. We will collect loss and accuracy metrics for comparison.


```python
dmalrc = create_model()
dmalrc_batch_hist = LossHistory()
metrics_deep_lrc_augmented = dmalrc.fit_generator(idg.flow(X_train,y_train, batch_size=batch_size),
                    epochs = epochs,
                    steps_per_epoch=X_train.shape[0]//batch_size,
                    validation_data=(X_cv,y_cv),
                    callbacks=[lrc,dmalrc_batch_hist],                         
                    verbose = verbose)
#dmalrc.save_weights("deep_lrc_augmented.h5")
```

    Epoch 1/20
    5s - loss: 0.4015 - acc: 0.8705 - val_loss: 0.0665 - val_acc: 0.9777
    Epoch 2/20
    5s - loss: 0.1398 - acc: 0.9575 - val_loss: 0.0580 - val_acc: 0.9812
    Epoch 3/20
    5s - loss: 0.1057 - acc: 0.9684 - val_loss: 0.0515 - val_acc: 0.9854
    Epoch 4/20
    5s - loss: 0.0934 - acc: 0.9714 - val_loss: 0.0422 - val_acc: 0.9870
    Epoch 5/20
    4s - loss: 0.0793 - acc: 0.9750 - val_loss: 0.0323 - val_acc: 0.9902
    Epoch 6/20
    5s - loss: 0.0731 - acc: 0.9777 - val_loss: 0.0448 - val_acc: 0.9886
    Epoch 7/20
    5s - loss: 0.0667 - acc: 0.9796 - val_loss: 0.0346 - val_acc: 0.9890
    Epoch 8/20
    5s - loss: 0.0640 - acc: 0.9809 - val_loss: 0.0276 - val_acc: 0.9917
    Epoch 9/20
    5s - loss: 0.0595 - acc: 0.9822 - val_loss: 0.0357 - val_acc: 0.9901
    Epoch 10/20
    5s - loss: 0.0548 - acc: 0.9831 - val_loss: 0.0301 - val_acc: 0.9917
    Epoch 11/20
    
    Epoch 00010: reducing learning rate to 0.0005000000237487257.
    5s - loss: 0.0536 - acc: 0.9832 - val_loss: 0.0313 - val_acc: 0.9912
    Epoch 12/20
    5s - loss: 0.0376 - acc: 0.9887 - val_loss: 0.0265 - val_acc: 0.9914
    Epoch 13/20
    5s - loss: 0.0352 - acc: 0.9901 - val_loss: 0.0285 - val_acc: 0.9913
    Epoch 14/20
    5s - loss: 0.0354 - acc: 0.9889 - val_loss: 0.0248 - val_acc: 0.9930
    Epoch 15/20
    5s - loss: 0.0337 - acc: 0.9893 - val_loss: 0.0264 - val_acc: 0.9929
    Epoch 16/20
    5s - loss: 0.0313 - acc: 0.9903 - val_loss: 0.0244 - val_acc: 0.9931
    Epoch 17/20
    
    Epoch 00016: reducing learning rate to 0.0002500000118743628.
    5s - loss: 0.0309 - acc: 0.9904 - val_loss: 0.0296 - val_acc: 0.9931
    Epoch 18/20
    5s - loss: 0.0274 - acc: 0.9913 - val_loss: 0.0237 - val_acc: 0.9933
    Epoch 19/20
    5s - loss: 0.0247 - acc: 0.9918 - val_loss: 0.0252 - val_acc: 0.9936
    Epoch 20/20
    5s - loss: 0.0273 - acc: 0.9915 - val_loss: 0.0256 - val_acc: 0.9932
    


```python
plot_losses(dmalrc_batch_hist)
```


![png](output_27_0.png)


## Results

Plotted below are the loss and accuracy metrics on training and validation data from the three models.


```python
plot_metrics(metrics_deep,"Convolutional Neural Network")
plot_metrics(metrics_deep_lrc,"CNN with Learning Rate Annealer\n")
plot_metrics(metrics_deep_lrc_augmented,"CNN with Annealer and Data Augmentation\n")
```


![png](output_29_0.png)



![png](output_29_1.png)



![png](output_29_2.png)


## Classification Accuracy

Here is a summary of how the three models performed in terms of training, validation and test accuracy.


```python
models = [dm, dmlrc, dmalrc]
metrics = [metrics_deep, metrics_deep_lrc, metrics_deep_lrc_augmented]
names = ["Convolutional Neural Network", "CNN + Learning Rate Annealing", "CNN + LR + Data Augmentation"
         ]
data = []
for i, m in enumerate(zip(names, metrics, models)):
    data.append([m[0], "{:0.2f}".format(m[1].history["acc"][-1]*100), "{:0.2f}".format(m[1].history["val_acc"][-1]*100), "{:0.2f}".format(m[2].evaluate(X_test, y_test, verbose=0)[1]*100)])

results = pd.DataFrame(data, columns=("Model","Training Accuracy","Validation Accuracy", "Test Accuracy"))
from IPython.display import display, HTML
display(HTML(results.to_html(index=False)))
plt.bar(np.arange(len(results["Model"].values)),results["Training Accuracy"].values.astype("float64"), 0.2, color="lightblue")
plt.bar(np.arange(len(results["Model"].values))+0.2,results["Validation Accuracy"].values.astype("float64"), 0.2, color="steelblue")
plt.bar(np.arange(len(results["Model"].values))+0.4,results["Test Accuracy"].values.astype("float64"), 0.2, color="navy")
plt.ylim(97, 100)
plt.xticks(np.arange(len(results["Model"].values))+0.2, ["CNN","CNN+LR", "CNN+LR+Aug"])
plt.legend(["Training","Validation", "Test"],loc="best")
g = plt.gca()
g.spines["top"].set_visible(False)
g.spines["right"].set_visible(False)
plt.title("Accuracy")
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Training Accuracy</th>
      <th>Validation Accuracy</th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Convolutional Neural Network</td>
      <td>99.59</td>
      <td>98.96</td>
      <td>99.15</td>
    </tr>
    <tr>
      <td>CNN + Learning Rate Annealing</td>
      <td>99.87</td>
      <td>99.18</td>
      <td>99.30</td>
    </tr>
    <tr>
      <td>CNN + LR + Data Augmentation</td>
      <td>99.15</td>
      <td>99.32</td>
      <td>99.45</td>
    </tr>
  </tbody>
</table>





    <matplotlib.text.Text at 0x18622626d68>




![png](output_31_2.png)


#### Work In Progress


```python

```


```python

```
