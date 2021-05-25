from PIL import Image
import os
import numpy as np

base = os.getcwd()
img_path = os.path.join(base, "imgs")

os.chdir(img_path)

filelist = os.listdir(img_path)

x = np.array([np.array(Image.open(fname)) for fname in filelist])
print(x)
print(x.dtype)


