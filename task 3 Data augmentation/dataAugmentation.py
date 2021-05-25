import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

base = "C:\Python proj\densen"
img_path = os.path.join(base, "imgs")
os.chdir(img_path)

filelist = os.listdir(img_path)
filenames = tf.constant(filelist[0:8])
dataset = tf.data.Dataset.from_tensor_slices((filenames[0:8]))

#step 3: parse every image in the dataset using `map`

def _parse_function(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded, tf.float32)
    return image

AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = dataset.map(_parse_function, AUTOTUNE)
dataset = dataset.batch(1)

def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        # output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        # row += 1
        print((images.numpy()[0][0]))

    plt.figure()
    plt.imshow(output)
    plt.show()

plot_images(dataset,8,10)
#
# def augment(x: tf.Tensor) -> tf.Tensor:
#     """Some augmentation
#
#     Args:
#         x: Image
#
#     Returns:
#         Augmented image
#     """
#     x = .... # augmentation here
#     return x