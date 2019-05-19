from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Reshape, Embedding,InputLayer

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    

#(x_train, _), (x_test, _) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
"""
x_train = x_train[:1000].astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train[:1000], (len(x_train[:1000]), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
y_train=y_train[:1000]

input_tensor = x_train.shape[1:]  # adapt this if using `channels_first` image data format

encoder_model = Sequential()
encoder_model.add(InputLayer(input_shape=input_tensor))
encoder_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.summary()
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

decoder_model = Sequential()
decoder_model.add(InputLayer(input_shape=encoder_model.output_shape[1:]))
decoder_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
decoder_model.add(UpSampling2D((2, 2)))
decoder_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
decoder_model.add(UpSampling2D((2, 2)))
decoder_model.add(Conv2D(16, (3, 3), activation='relu'))
decoder_model.add(UpSampling2D((2, 2)))
decoder_model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
decoder_model.summary()

autoencoder = Model(input=encoder_model.input, output=decoder_model(encoder_model.output))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

#tensorboard --logdir=/tmp/autoencoder

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(
    log_dir = './tmp/autoencoder',
    histogram_freq=0,
    batch_size=32,
    write_graph=True,
    write_grads=True,
    write_images=False,
)
callbacks=[tensorboard]
autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=callbacks
                )

decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder_model.predict(x_test)

for j in range(10):
    x_train1 = x_train[y_train==1]
    x_test1 = x_test[y_test==1]
   
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
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test1, x_test1)
                #callbacks=callbacks
                )

    decoded_imgs = autoencoder.predict(x_test)
    encoded_imgs = encoder_model.predict(x_test)
    decoder_imgs = decoder_model.predict(encoded_imgs) 
        
    #plot_fig(x_test,decoded_imgs,encoded_imgs,j)
    #plot_fig_decoder(decoded_imgs,decoder_imgs,j)

# 実行結果の表示
n = 10
decoded_imgs = autoencoder.predict(x_test[:n])
image_size=28

plt.figure(figsize=(10, 4))
for i in range(n):
    # original_image
    orig_img = x_test[i].reshape(image_size, image_size)

    # reconstructed_image
    reconst_img = decoded_imgs[i].reshape(image_size, image_size)

    # diff image
    diff_img = ((orig_img - reconst_img)+2)/4
    diff_img = (diff_img*255).astype(np.uint8)
    orig_img = (orig_img*255).astype(np.uint8)
    reconst_img = (reconst_img*255).astype(np.uint8)
    diff_img_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)

    # display original
    ax = plt.subplot(3, n,  i + 1)
    plt.imshow(orig_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(reconst_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display diff
    ax = plt.subplot(3, n, i + n*2 + 1)
    plt.imshow(diff_img, cmap=plt.cm.jet)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    