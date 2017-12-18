import os
import glob
import numpy as np
import cv2
import pickle
from sklearn.utils import shuffle


def load_images(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            try:
                image = cv2.imread(fl)
                image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                ids.append(flbase)
                cls.append(fld)
            except:
                continue
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

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


def read_train_sets(train_path, image_size, classes, validation_size=0, state_dir='', from_file_if_saved=True):
    if from_file_if_saved and state_dir != '':
        if os.path.isfile(state_dir + "train.p") and os.path.isfile(state_dir + "valid.p"):
            return pickle.load(open(state_dir + "train.p", "rb")), pickle.load(open(state_dir + "valid.p", "rb"))

    images, labels, ids, cls = load_images(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

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

    if state_dir != '':
        pickle.dump(train, open(state_dir + "train.p", "wb"))
        pickle.dump(valid, open(state_dir + "valid.p", "wb"))

    return train, valid


def read_test_set(test_path, image_size, classes, state_dir='', from_file_if_saved=True):
    if from_file_if_saved and state_dir != '':
        if os.path.isfile(state_dir + "test.p"):
            return pickle.load(open(state_dir + "test.p", "rb"))
    images, labels, ids, cls = load_images(test_path, image_size, classes)
    test = DataSet(images, labels, ids, cls)
    if state_dir != '':
        pickle.dump(test, open(state_dir + "test.p", "wb"))
    return test
