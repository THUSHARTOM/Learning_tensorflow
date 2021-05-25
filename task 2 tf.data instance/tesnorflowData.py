import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

base = os.getcwd()
img_path = os.path.join(base, "imgs")
os.chdir(img_path)

filelist = os.listdir(img_path)
labellist=[]
print(filelist)
for count,ele in enumerate(filelist):
    if count%2 ==0:
        labellist.append(1)
    else:
        labellist.append(0)

filenames = tf.constant(filelist)
labels = tf.constant(labellist)

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
print(dataset)

#step 3: parse every image in the dataset using `map`

def _parse_function(filename, label):
    image_string = tf.compat.v1.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label
AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = dataset.map(_parse_function, AUTOTUNE)
dataset = dataset.batch(1)

# step 4: create iterator and final input tensor
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
for _ in range(len(filelist)-1):
    image, label = iterator.get_next()
    img_array = image.numpy()[0]
    print(img_array)
    height, width, _ = img_array.shape
    print(height, width)

    for i in range(0, height - 1):
        for j in range(0, width - 1):
            # Get the pixel value
            pixel = img_array[i, j]

            # Negate each channel by
            # subtracting it from 255

            # 1st index contains red pixel
            pixel[0] = 255 - pixel[0]

            # 2nd index contains green pixel
            pixel[1] = 255 - pixel[1]

            # 3rd index contains blue pixel
            pixel[2] = 255 - pixel[2]

            # Store new values in the pixel
            img_array[i, j] = pixel

    # Display the negative transformed image
    plt.imshow((img_array).astype(np.uint8))
    plt.show()