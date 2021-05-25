import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

base = "C:\Python proj\densen"
img_path = os.path.join(base, "imgs")
os.chdir(img_path)

filelist = os.listdir(img_path)
filenames = tf.constant(filelist)
dataset = tf.data.Dataset.from_tensor_slices((filenames))

#step 3: parse every image in the dataset using `map`

def _parse_function(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded, tf.float32)
    return image

AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = dataset.map(_parse_function, AUTOTUNE)
dataset = dataset.batch(1)

for i in dataset:

    i = np.array([255.0]) - i
    plt.imshow((i[0].numpy()).astype(np.uint8))
    plt.show()
    # print(i)