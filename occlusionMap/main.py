from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

imgOg = Image.open('car.jpg')
imgOg = imgOg.resize((64, 64))
#imgOg.save('og.jpg')
pixelMapOg = imgOg.load()

boxSize = 24
z = 0
for k in range(64-boxSize-1):
  for m in range(64-boxSize-1):
    imgNew = imgOg.copy()
    pixelsNew = imgNew.load()

    for i in range(k, k+boxSize):    # for every col:
      for j in range(m, m+boxSize):    # For every row
        pixelsNew[i,j] = 0 
    
    imgNew.save('48_IMG_' + str(z) + '.jpg')
    z+=1

#imgOg.save('finalOG.jpg')
print('done')
