import time
import math
import random

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data_processing as dp
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['Swamp Milkweed', 'Common Milkweed', 'Yellow Wild', 'Partridge Pea', 'Showy Tick', 'Common Boneset',
           'Joe-Pye Weed', 'Sunflower', 'Round-head Bush-Clover', 'Dense Blazing-star', 'Shaggy Blazing-star',
           'Great Blue', 'Wild Blue', 'Wild Bergamot', 'Spotted Horsemint', 'Beardtongue', 'Bigleaf Mountain Mint',
           'Common Mountain Mint', 'Black-Eyed Susan', 'Coneflower', 'Canada Goldenrod', 'Early Goldenrod',
           'New England Aster', 'Hairy Heath Aster', 'Spiderworts', 'Blue Vervain', 'New York Ironweed',
           'Golden Alexanders', 'Big Bluestem', 'Switch Grass', 'Little Bluestem', 'Indian Grass', 'Eastern Gamagrass',
           'Canada Wild-rye', 'Virginia Wild-rye', 'Red Maple', 'Black Cherry', 'Allegheny Serviceberry', 'Buttonbush',
           'Winterberry Holly', 'Pawpaw', 'Virginia Sweetspire', 'Flowering dogwood', 'Silky dogwood', 'Redbud',
           'Mountain Laurel', 'Witchhazel', 'Spicebush', 'Arrowwood viburnum', 'Blackhaw viburnum',
           'Highbush Blueberry', 'Bush Honeysuckles', 'Butterfly Bush', 'Lesser Celandine', 'Porcelainberry',
           'Bradford Pear', 'Oriental Bittersweet', 'Beefsteak Plant', 'Japanese Barberry', 'Wisteria',
           'Chocolate Vine', 'Periwinkle', 'Common Daylily', 'Spanish Bluebells', 'Winged Burning Bush', 'Norway Maple',
           'Wintercreeper', 'Chinese Silvergrass', 'Doublefile Viburnum', 'Italian Arum', 'Jetbead',
           'Leatherleaf Mahonia', 'Heavenly Bamboo']
num_classes = len(classes)

# batch size
batch_size = 28

# validation split
validation_size = .16

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

total_iterations = 0

train_path = 'data/train/'
test_path = 'data/test/'
state_dir = "data/"

train, valid = dp.read_train_sets(train_path, img_size, classes, validation_size=validation_size, state_dir=state_dir)
test = dp.read_test_set(test_path, img_size, classes, state_dir=state_dir)



print("Size of:")
print("- Training-set:\t\t{}".format(len(train.labels)))
print("- Test-set:\t\t\t{}".format(len(test.labels)))
print("- Validation-set:\t{}".format(len(valid.labels)))


class CNN():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

        self.x_image = tf.reshape(self.x, [-1, img_size, img_size, num_channels])

        self.y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

        self.y_true_cls = tf.argmax(self.y_true, axis=1)

        self.layer_conv1, self.weights_conv1 = \
            self.new_conv_layer(input=self.x_image,
                                num_input_channels=num_channels,
                                filter_size=filter_size1,
                                num_filters=num_filters1,
                                use_pooling=True)

        self.layer_conv2, self.weights_conv2 = \
            self.new_conv_layer(input=self.layer_conv1,
                                num_input_channels=num_filters1,
                                filter_size=filter_size2,
                                num_filters=num_filters2,
                                use_pooling=True)

        self.layer_conv3, self.weights_conv3 = \
            self.new_conv_layer(input=self.layer_conv2,
                                num_input_channels=num_filters2,
                                filter_size=filter_size3,
                                num_filters=num_filters3,
                                use_pooling=True)

        self.layer_flat, self.num_features = self.flatten_layer(self.layer_conv3)

        self.layer_fc1 = self.new_fc_layer(input=self.layer_flat,
                                           num_inputs=self.num_features,
                                           num_outputs=fc_size,
                                           use_relu=True)

        self.layer_fc2 = self.new_fc_layer(input=self.layer_fc1,
                                           num_inputs=fc_size,
                                           num_outputs=num_classes,
                                           use_relu=False)

        self.y_pred = tf.nn.softmax(self.layer_fc2)

        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layer_fc2, labels=self.y_true)

        self.cost = tf.reduce_mean(self.cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)

        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver(max_to_keep=10)
        self.session = tf.InteractiveSession()

        # session.run(tf.global_variables_initializer())
        self.new_saver = tf.train.import_meta_graph(state_dir + 'cnn.meta')
        self.new_saver.restore(self.session, tf.train.latest_checkpoint(state_dir))

        self.train_batch_size = batch_size

    def plot_images(self, images, cls_true, cls_pred=None):
        if len(images) == 0:
            print("no images to show")
            return
        else:
            random_indices = random.sample(range(len(images)), min(len(images), 9))

        images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])

        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(img_size, img_size, num_channels))

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(self,
                       input,              # The previous layer.
                       num_input_channels, # Num. channels in prev. layer.
                       filter_size,        # Width and height of each filter.
                       num_filters,        # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    def flatten_layer(self, layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(self,
                     input,          # The previous layer.
                     num_inputs,     # Num. inputs from prev. layer.
                     num_outputs,    # Num. outputs.
                     use_relu=True): # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def print_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        # Calculate the accuracy on the training-set.
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))

    def optimize(self, num_iterations):
        # Ensure we update the global variable rather than a local copy.
        global total_iterations

        # Start-time used for printing time-usage below.
        start_time = time.time()

        best_val_loss = float("inf")
        patience = 0

        for i in range(total_iterations,
                       total_iterations + num_iterations):

            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch, _, cls_batch = train.next_batch(self.train_batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = valid.next_batch(self.train_batch_size)

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, flattened image shape]

            x_batch = x_batch.reshape(self.train_batch_size, img_size_flat)
            x_valid_batch = x_valid_batch.reshape(self.train_batch_size, img_size_flat)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            feed_dict_validate = {self.x: x_valid_batch,
                                  self.y_true: y_valid_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.session.run(self.optimizer, feed_dict=feed_dict_train)

            # Print status at end of each epoch (defined as full pass through training dataset).
            if i % int(train.num_examples / batch_size) == 0:
                self.saver.save(self.session, state_dir + 'cnn')
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_validate)
                epoch = int(i / int(train.num_examples / batch_size))

                self.print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience = 0
                    else:
                        patience += 1

                    if patience == early_stopping:
                        break

        # Update the total number of iterations performed.
        total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

    def plot_example_errors(self, cls_pred, correct):
        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = valid.images[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = valid.cls[incorrect]

        # Plot the first 9 images.
        self.plot_images(images=images[0:9],
                         cls_true=cls_true[0:9],
                         cls_pred=cls_pred[0:9])

    def plot_confusion_matrix(self, cls_pred):
        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the true classifications for the test-set.
        cls_true = valid.cls

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=cls_pred)

        # Print the confusion matrix as text.
        print(cm)

        # Plot the confusion matrix as an image.
        plt.matshow(cm)

        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def print_validation_accuracy(self,
                                  show_example_errors=False,
                                  show_confusion_matrix=False):
        # Number of images in the test-set.
        num_test = len(valid.images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_test)

            # Get the images from the test-set between index i and j.
            to_size = valid.images[i:j, :].length() / img_size_flat
            images = valid.images[i:j, :].reshape(to_size, img_size_flat)

            # Get the associated labels.
            labels = valid.labels[i:j, :]

            # Create a feed-dict with these images and labels.
            feed_dict = {self.x: images,
                         self.y_true: labels}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = self.session.run(self.y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        cls_true = np.array(valid.cls)
        cls_pred = np.array([classes[x] for x in cls_pred])

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            self.plot_example_errors(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            self.plot_confusion_matrix(cls_pred=cls_pred)

cnn=CNN()

cnn.optimize(num_iterations=1000)

#print_validation_accuracy()
