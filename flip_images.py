import math
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

filelist_train = glob.glob('data/rawImages/car_ims/*9.jpg')

data_train = np.array([np.array(Image.open(fname)) for fname in filelist_train])

for fname in filelist_train:
  im = Image.open(fname)
  fliped_im = im.transpose(Image.FLIP_LEFT_RIGHT)
  fliped_im.save('data/rawImages/car_ims/1'+fname.split('/')[-1])
  #liped_im.show()

