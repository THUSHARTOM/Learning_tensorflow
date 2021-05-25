import numpy as np
from PIL import Image
import cv2
import os

base = os.getcwd()
img_path = os.path.join(base, "imgs")
os.chdir(img_path)

filelist = os.listdir(img_path)

im = np.array(Image.open(img_path + os.sep + filelist[1]))
size_x = im.shape[0] + 100
size_y = im.shape[1]*2 + 200

black_img = np.zeros((size_x, size_y, 3), dtype="uint8")

background_img = Image.fromarray(black_img)
true_img = Image.fromarray(im)
rotated_180_img = Image.fromarray(np.rot90(im, 2))

print(rotated_180_img.size)
print(background_img.size)
print(true_img.size)

# rotated_180_img.show()

background_img.paste(true_img, (50,50))
background_img.paste(rotated_180_img, (int(size_y/2)+50,50))
background_img.show()