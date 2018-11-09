# Imports
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# Import image to class mapping csv file
with open('data/image_class_labels.csv', 'r') as content_file:
	labelContent = content_file.read()
classArray = labelContent.split('\n')
classArray[0] = 1
del classArray[-1]

# Import class to label mapping
with open('data/class_names.csv', 'r') as content_file:
	content = content_file.read()
namesArray = content.split(',')
datesArray = []
brandArray = []
# Create arrays of pertinent data from class to label mapping
for i in namesArray:
	datesArray.append(i.split(' ')[-1].split('\'')[0])
	brandArray.append(i.split(' ')[0].split('\'')[-1])	

# Plot frequency analysis of brands and years to help determine bins
brand_dict = {}
date_dict = {}
for class_i in classArray:
	name_i = brandArray[int(class_i)-1]
	date_i = datesArray[int(class_i)-1]
	if name_i in brand_dict:
		brand_dict[name_i] += 1
	else:
		brand_dict[name_i] = 1
	if date_i in date_dict:
		date_dict[date_i] += 1
	else:
		date_dict[date_i] = 1
# Brand plotting
brand_dict = OrderedDict(sorted(brand_dict.items(), key = lambda t: t[1]))
brand_x_labels = list(brand_dict.keys())
brand_y_vals = list(brand_dict.values())
plt.bar(range(len(brand_dict)), brand_y_vals, tick_label = brand_x_labels)
plt.xticks(fontsize = 4)
plt.show()
# Year plotting
date_dict = OrderedDict(sorted(date_dict.items()))
date_x_labels = list(date_dict.keys())
date_y_vals = list(date_dict.values())
plt.bar(range(len(date_dict)), date_y_vals, tick_label = date_x_labels)
plt.show()

# Relabel brand_dict to separate out all classes
cnt = 0
for key in brand_dict:
	brand_dict[key] = cnt
	cnt += 1

# Create array of bins by dates
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

# Create array of bins by make/brand
binsArrayNames = []
for i in brandArray:
	binsArrayNames.append(brand_dict[i])	

# Relabel image filenames to include classification information (brand)
filelist_data = glob.glob('data/rawImages/car_ims/*.jpg')
filelist_data.sort()
cnt = 0
for fname in filelist_data:
	curImage = Image.open(fname)
	filesName = fname.split('/')[-1]   # .split('.')[0]
	newName = str(binsArrayNames[int(classArray[cnt])-1]) + '_IMG_' + filesName
	cnt += 1
	curImage.save('data/renamedImages/' + newName)

