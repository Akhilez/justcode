import numpy as np
import os
from PIL import Image
from resizeimage import resizeimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from keras.layers import Conv2D, InputLayer, UpSampling2D
from keras.models import Sequential
from keras.models import load_model
from skimage.io import imsave

print("imported successfully")

def resize_image(img, width, height):
    with open(img, 'r+b') as f_image:
        with Image.open(f_image) as image:
            cover = resizeimage.resize_cover(image, [width, height])
            cover.save(img, image.format)

def preprocess_img(image_path):
    print ("image under preprocess: "+image_path)

    # saving the image as a 256x256 image
    resize_image(image_path, 256, 256)

    # get the 256x256x3 list from the image
    # Each pixel has values of R, G and B
    # Each value ranges [0, 255]
    image = img_to_array(load_img(image_path))

    # Converting the list into a numpy array for each computations
    image = np.array(image, dtype=float)

    # Changing the range from [0, 255] to [0, 1] for each R, G, B value
    image = image / 255

    # Converting each pixel from [R, G, B] to [L, a, b] range [0,1]
    image = rgb2lab(image) / 100

    return image

def get_model():

    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    return model


def preprocess_dir (images_path):
    #return np.array([preprocess_img(images_path+'/'+image_name) for image_name in os.listdir(images_path)])

    images = []

    for image_name in os.listdir(images_path):
        images.append(preprocess_img(images_path+'/'+image_name))

    images = np.array(images)

    return images

def batch_generator (images, batch_size):
    image_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True
    )
    for batch in image_generator.flow(images, batch_size=batch_size):
        # Extract L where each pixel is [L] range [0,1] shape [num_images, 256, 256, 1]
        L = batch[:, :, :, :1]
        # Extract ab where each pixel is [a, b] range [-1, 1] shape [num_images, 256, 256, 2]
        ab = batch[:, :, :, 1:]

        yield L, ab

def process(images_path, train=False):
    images = preprocess_dir(images_path)

    if train:
        train_new_model(images)
    else:
        model = get_trained_model()
        postprocess(images, model)

def train_new_model (images):

    batch_size = 2
    num_images = len(images)
    steps_per_epoch = int(num_images / batch_size)

    batch_generator_object = batch_generator (images, batch_size)

    model = get_model ()
    model.fit_generator(batch_generator_object, epochs=10, steps_per_epoch=steps_per_epoch)

    model.save("models/model.h5")


def get_trained_model ():
    model = load_model("models/model.h5")
    return model

def postprocess(images, model):
    L = images[:, :, :, :1]
    ab = images[:, :, :, 1:]

    predicted_ab = model.predict(L)
    error = ab - predicted_ab

    print("Max error = " + str(np.max(error)))
    print("Min error = " + str(np.min(error)))

    for i in range(len(images)):
        color_image = np.zeros(shape=(256, 256, 3))
        color_image[:, :, :1] = L[i]
        color_image[:, :, 1:] = predicted_ab[i]
        color_image = color_image * 100

        color_image = lab2rgb(color_image)

        imsave("results/img_"+str(i)+".jpg", color_image)


process("Train", train=True)
process("Test", train=False)
