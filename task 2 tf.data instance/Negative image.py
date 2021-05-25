import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

base = os.getcwd()
img_path = os.path.join(base, "imgs")
os.chdir(img_path)

filelist = os.listdir(img_path)

im = np.array(Image.open(img_path + os.sep + filelist[1]))
print(im)
img_bgr =im
# get height and width of the image
height, width, _ = im.shape
print(height,width)

for i in range(0, height - 1):
    for j in range(0, width - 1):
        # Get the pixel value
        pixel = img_bgr[i, j]

        # Negate each channel by
        # subtracting it from 255

        # 1st index contains red pixel
        pixel[0] = 255 - pixel[0]

        # 2nd index contains green pixel
        pixel[1] = 255 - pixel[1]

        # 3rd index contains blue pixel
        pixel[2] = 255 - pixel[2]

        # Store new values in the pixel
        img_bgr[i, j] = pixel

# Display the negative transformed image
plt.imshow(img_bgr)
plt.show()
