import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

with open('data/image_class_labels.csv', 'r') as content_file:
	labelContent = content_file.read()
classArray = labelContent.split('\r\n')
classArray[0] = 1


with open('data/class_names.csv', 'r') as content_file:
	content = content_file.read()
namesArray = content.split(',')
datesArray = []
for i in namesArray:
	datesArray.append(i.split(' ')[-1].split('\'')[0])

binsArray = []
for i in datesArray:
	if (int(i) >= 1991) and (int(i) <= 2004):
		binsArray.append(0)
	elif (int(i) >= 2005) and (int(i) <=2007):
		binsArray.append(1) 
	elif (int(i) >= 2008) and (int(i) <=209):
		binsArray.append(2)
	elif (int(i) >= 2010) and (int(i) <=2010):
		binsArray.append(3)
	elif (int(i) >= 2011) and (int(i) <=2011):
		binsArray.append(4)
	elif (int(i) >= 2012) and (int(i) <=2012):
		binsArray.append(5)

datesArrayInts = list(map(int,datesArray))
plt.hist(datesArrayInts)
plt.show()



filelist_data = glob.glob('data/rawImages/car_ims/*.jpg')
cnt = 0
for fname in filelist_data:
	curImage = Image.open(fname)
	filesName = fname.split('/')[-1]   # .split('.')[0]
	newName = str(binsArray[int(classArray[cnt])-1]) + '_IMG_' + filesName
	cnt += 1
	curImage.save('data/renamedImages/' + newName)

