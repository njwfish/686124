from os import walk, makedirs
from shutil import copyfile
from random import shuffle
from math import floor

for (dirpath, dirnames, filenames) in walk('/Users/njwfish/Dropbox/edrr/classifier/plants/'):
	curr_label = dirpath.split('/')[-1]
	test_dir = '/Users/njwfish/Dropbox/edrr/classifier/plants/test/' + curr_label
	train_dir = '/Users/njwfish/Dropbox/edrr/classifier/plants/train/' + curr_label
	makedirs(test_dir)
	makedirs(train_dir)
	n = len(filenames)
	shuffle(filenames)
	for file in filenames[:floor(n*0.2)]:
		copyfile(dirpath + '/' + file, test_dir + '/' + file)
	for file in filenames[floor(n*0.2)+1:]:
		copyfile(dirpath + '/' + file, train_dir + '/' + file)

