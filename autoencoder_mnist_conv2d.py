from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Reshape, Embedding,InputLayer



def plot_fig_decoder(decoded_imgs, encoder_imgs,j):
    n = 100
    plt.figure(figsize=(10, 16))
    for i in range(1,n):
        # display original
        ax = plt.subplot(20, n*0.1, i)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(20, n*0.1, i + n)
        plt.imshow(encoder_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("mnist_decoded_encoder_by_100_10_{}".format(j))    
    plt.pause(0.01)
    plt.close()

def plot_fig(x_test, decoded_imgs, encoded_imgs,j):
    n = 100
    plt.figure(figsize=(10, 16))
    for i in range(1,n):
        # display original
        ax = plt.subplot(20, n*0.1, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(20, n*0.1, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("mnist_training_by_100_10_{}".format(j))    
    plt.pause(0.01)
    plt.close()

    n = 100
    plt.figure(figsize=(10, 16))
    for i in range(1,n):
        ax = plt.subplot(10, n*0.1, i)
        plt.imshow(encoded_imgs[i].reshape(8, 2 * 8).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("mnist_intermid_training_by_100_10_{}".format(j))    
    plt.pause(0.01)
    plt.close()

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same',name='encoded')(x)
encoder=Model(input_img, encoded)
encoder.summary()

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

from keras.datasets import mnist
import numpy as np

#(x_train, _), (x_test, _) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:1000].astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train[:1000], (len(x_train[:1000]), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
y_train=y_train[:1000]

for j in range(10):
    x_train1 = x_train  #[y_train==3]
    x_test1 = x_test #[y_test==3]
   
    tensorboard = TensorBoard(
        log_dir = './tmp/autoencoder',
        histogram_freq=0,
        batch_size=16,
        write_graph=True,
        write_grads=True,
        write_images=False,
    )
    #callbacks=[tensorboard]
    autoencoder.fit(x_train1, x_train1,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test1, x_test1)
                #callbacks=callbacks
                )

    decoded_imgs = autoencoder.predict(x_test)
    encoded_imgs = encoder.predict(x_test)
    decoder_imgs = decoder.predict(encoded_imgs) 
        
    plot_fig(x_test,decoded_imgs,encoded_imgs,j)
    plot_fig_decoder(decoded_imgs,decoder_imgs)
   

