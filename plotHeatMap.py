# Imports
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

with open('heatMapRaw.txt') as f:
	content = f.readlines()

floatValues = []
for i in content:
	floatValues.append(float(i.split(' ')[-1].strip()))

rawImage = np.asarray(floatValues).reshape((39,39)).T
plt.imshow(rawImage,cmap='hot',interpolation='quadric')
plt.show()

