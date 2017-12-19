#!/usr/local/bin/python3.6

from os import path, walk, makedirs
from shutil import copyfile, rmtree
from random import shuffle
from math import floor
import image_search
import glob
import numpy as np
import cv2
import pickle
from sklearn.utils import shuffle


class GoogleImagesDataSet:
    """Takes a list of classes in a directory, 'classes.txt', and for each class downloads images from Google Images,
        into directories, building an image classification dataset. Splits class folders into testing and
        training sets."""
    def __init__(self, data_dir, examples_per_class, test_size):
        """
        Creates ImageDataSet object, use get_classes() to actually create dataset
        :param data_dir: directory to create class test and train folders, also location of 'classes.txt'
        :param examples_per_class: number of images per class
        :param test_size: size of the test set, float between 0 and 1
        """
        self.data_dir = data_dir
        self.examples_per_class = examples_per_class
        self.test_size = test_size
        self.classes = self.get_classes()

    def get_classes(self):
        """Uses 'classes.txt' to read list of classes. Each class should be the exact query you want to search Google
           for, and each class should be on a new line"""
        with open(self.data_dir + 'classes.txt') as f:
            content = f.readlines()
        return [x.strip().lower() for x in content]

    def gen_dataset(self):
        """Takes a list of classes in a directory, 'classes.txt', and for each class downloads images from Google
           Images, into directories, building an image classification dataset. Splits class folders into testing and
           training sets."""
        self.get_images()
        self.split_train_test()

    def get_images(self):
        """Use Selenium Image Search to get images"""
        sel = image_search.SeleniumImageSearch()
        num_results = self.examples_per_class
        for cls in self.classes:
            cls_directory = self.data_dir + cls + '/'
            if not path.exists(cls_directory):
                makedirs(cls_directory)
            sel.search(cls.lower(), cls_directory, num_results)

    def split_train_test(self):
        """Randomly splits all classes into a test set and a training set, in the folders test and training respectively
           deleting original class folders as it goes."""
        for (dirpath, dirnames, filenames) in walk(self.data_dir):
            if dirpath == self.data_dir:
                continue
            curr_label = dirpath.split('/')[-1]
            test_dir = self.data_dir + 'test/' + curr_label
            train_dir = self.data_dir + 'train/' + curr_label
            makedirs(test_dir)
            makedirs(train_dir)
            n = len(filenames)
            shuffle(filenames)
            for file in filenames[:floor(n * self.test_size)]:
                copyfile(dirpath + '/' + file, test_dir + '/' + file)
            for file in filenames[floor(n * self.test_size) + 1:]:
                copyfile(dirpath + '/' + file, train_dir + '/' + file)
            rmtree(dirpath)


class ImageDataSet:
    def __init__(self, state_dir, image_size, classes, validation_size):
        """Assumes the data is structured from a GoogleImageDataSet, reads and formats images for use in CNN image
           classifacation model. Saves dataset objects after loading to prevent having to load each time.
        :param data_dir: directory containing test, train
        :param classes: list of classes
        """
        self.state_dir = state_dir
        self.image_size = image_size
        self.classes = classes
        self.validation_size = validation_size
        self.train_path = self.state_dir + 'train/'
        self.test_path = self.state_dir + 'test/'

    def read_train_sets(self, from_file_if_saved=True):
        if from_file_if_saved and self.state_dir != '':
            if path.isfile(self.state_dir + "train.p") and path.isfile(self.state_dir + "valid.p"):
                return pickle.load(open(self.state_dir + "train.p", "rb")), \
                       pickle.load(open(self.state_dir + "valid.p", "rb"))

        images, labels, ids, cls = self.load_images(self.train_path, self.image_size, self.classes)
        images, labels, ids, cls = shuffle(images, labels, ids, cls)

        validation_size = int(self.validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_ids = ids[:validation_size]
        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_ids = ids[validation_size:]
        train_cls = cls[validation_size:]

        train = DataSet(train_images, train_labels, train_ids, train_cls)
        valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

        pickle.dump(train, open(self.state_dir + "train.p", "wb"))
        pickle.dump(valid, open(self.state_dir + "valid.p", "wb"))

        return train, valid

    def read_test_set(self, from_file_if_saved=True):
        if from_file_if_saved:
            if path.isfile(self.state_dir + "test.p"):
                return pickle.load(open(self.state_dir + "test.p", "rb"))
        images, labels, ids, cls = self.load_images(self.test_path, self.image_size, self.classes)
        test = DataSet(images, labels, ids, cls)
        pickle.dump(test, open(self.state_dir + "test.p", "wb"))
        return test

    @staticmethod
    def load_images(data_path, image_size, classes):
        """Assumes data_path is structured as a set of folders, each representing a class, with images of that class.
           This is the folder architecture GoogleImageDataSet creates when generating an image dataset. As a processing
           step this resizes all images to squares of length image_size, so NN can easily handle all images.
        :param data_path: folder containing class folders
        :param image_size:
        :param classes:
        :return: loaded images, the appropriate labels, the associated ids, and the list of classes
        """
        images, labels, ids, cls = [], [], [], []
        print('Reading training images')
        for fld in classes:
            index = classes.index(fld)
            print('Loading {} files (Index: {})'.format(fld, index))
            image_path = path.join(data_path, fld, '*g')
            files = glob.glob(image_path)
            for fl in files:
                # For some reason OpenCV doesn't like to resize some images. If an image can't be resized, skip it.
                try:
                    image = cv2.imread(fl)
                    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
                    images.append(image)
                    label = np.zeros(len(classes))
                    label[index] = 1.0
                    labels.append(label)
                    flbase = path.basename(fl)
                    ids.append(flbase)
                    cls.append(fld)
                except:
                    continue
        images, labels, ids, cls = np.array(images), np.array(labels), np.array(ids), np.array(cls)
        return images, labels, ids, cls


class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # Convert from [0, 255] -> [0.0, 1.0].

        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # # Shuffle the data (maybe)
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]

