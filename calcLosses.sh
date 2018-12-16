#!/bin/bash
for i in {0..1520} 
do	
	cd occlusionMap/images
	mv 48_IMG_$i.jpg ../test_signs/
	cd ../..
	python evaluate.py --data_dir occlusionMap --model_dir experiments/base_model &> temp.txt
	grep -oh "loss: ....." temp.txt >> heatMapRaw.txt
	rm occlusionMap/test_signs/*
	echo $i
done
