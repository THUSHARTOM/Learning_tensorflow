import numpy as np
from PIL import Image
import os

for filename in os.listdir("imgs"):
    print(filename)

    im = np.array(Image.open("imgs//" + filename))
    im_array = np.append(im,)
# print(type(im))
# print(im.dtype)
# print(im.shape)
