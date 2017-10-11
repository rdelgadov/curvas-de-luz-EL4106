from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
import tensorflow as tf

# Parámetros de entrenamiento
minibatch_size = 32
max_epochs = 100

# Parámetros de gradiente descendente estocastico
learning_rate = 0.1

# Número de neuronas
n_inputs = 1
n_hidden = 30
n_classes = 3
n_neurons = (n_inputs, n_hidden, n_classes)

#load Database, we need to create a loader.

def process_dataset(images, labels, selected_class):
    with open('db/Training.csv','rb') as csvfile:
        reader = csv.DictReader(csvfile)
        
    shuffled_indexes = np.random.permutation(len(labels))
    images = images[shuffled_indexes]
    labels = labels[shuffled_indexes]
    selected_column = labels[:, selected_class]
    selected_images_indexes = np.where(selected_column == 1)[0]
    selected_size = len(selected_images_indexes)
    non_selected_indexes_subset = np.where(selected_column == 0)[0][:selected_size]
    indexes = np.concatenate(
        (selected_images_indexes,
        non_selected_indexes_subset),
        axis=0
    )
    np.random.shuffle(indexes)
    images_subset = images[indexes]
    labels_subset = selected_column[indexes]
    return images_subset, labels_subset

training_images, training_labels = process_dataset(
    mnist.train.images,
    mnist.train.labels,
    RUT_veri_number
)

# Visualizar algunas imagenes del dataset
if False:
    for image, digit in zip(training_images[:5], training_labels[:5]):
        print("Etiqueta: %d" % int(digit))
        plt.imshow(image.reshape((28, 28)))
        plt.show()

validation_images, validation_labels = process_dataset(
    mnist.validation.images,
    mnist.validation.labels,
    RUT_veri_number
)

testing_images, testing_labels = process_dataset(
    mnist.test.images,
    mnist.test.labels,
    RUT_veri_number
)



SUMMARIES_DIR = './summaries'

# create MLP
MLP_input = tf.placeholder(tf.float32, shape=[None, n_inputs], name='MLP_input_placeholder')
previous_layer = MLP_input
# hidden layer creation
for level in range(len(n_neurons)-1):
    with tf.variable_scope("layer_"+str(level)):
        #weights = tf.get_variable("weights_"+str(level), [n_neurons[level],n_neurons[level+1]], initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.Variable(tf.truncated_normal([n_neurons[level],n_neurons[level+1]], stddev=0.1))
        biases = tf.get_variable("biases_"+str(level), [n_neurons[level+1]], initializer=tf.constant_initializer(0.0))
        applied_weights = tf.matmul(previous_layer, weights) + biases

        regularizers += tf.nn.l2_loss(weights)

        if level < len(n_neurons)-2:
            layer = tf.nn.sigmoid(applied_weights)
            previous_layer = layer
        else:
            MLP = applied_weights
            break

# Cost function
target = tf.placeholder(tf.float32, shape=None, name='target_placeholder')
one_hot_target = tf.one_hot(tf.cast(target, dtype=tf.int32), 2)

loss_function_name = 'cross_entropy'  # Choose between 'cross_entropy' or 'mse'

with tf.variable_scope("loss_function"):
    if loss_function_name == 'cross_entropy':
        # Cross Entropy
        loss_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=MLP,
                labels=one_hot_target,
                name='loss')
        )
    elif loss_function_name == 'mse':
        # Mean Squared Error
        loss_function = tf.reduce_mean(tf.square(one_hot_target - tf.nn.softmax(logits=MLP)))
    else:
        raise ValueError('Wrong value for loss_function_name')

with tf.variable_scope("accuracy"):

    correct_predictions = tf.equal(tf.argmax(MLP, 1),
                                   tf.argmax(one_hot_target, 1))

    accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, tf.float32), name='accuracy')
# Tensorboard summary
acc_sum = tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("normal/moving_mean",weights)
# loss function summary
loss_summ = tf.summary.scalar("loss_function", loss_function)

# add all summaries to summ
summ = tf.summary.merge_all()
val_summ = tf.summary.merge([acc_sum, loss_summ])

# Optimization method
with tf.variable_scope("train_process"):
    # Gradient Descent with Momentum
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Adam Algorithm
    #optimizer = tf.train.AdamOptimizer()

    training_algorithm = optimizer.minimize(loss_function)

#model training
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

writer1 = tf.summary.FileWriter(logdir+"/train")
writer2 = tf.summary.FileWriter(logdir+"/validation")
writer3 = tf.summary.FileWriter(logdir+"/weight")
writer1.add_graph(sess.graph)

n_minibatches = int(np.shape(training_images)[0] / minibatch_size)

prev_validation_loss = 100.0
validation_checks = 0
max_validation_checks = 1000
validation_period = 10

TRAIN_STOP_FLAG = False

for epoch in range(max_epochs):
    if TRAIN_STOP_FLAG:
        break
    new_indexes = np.random.permutation(len(training_labels))
    training_images[new_indexes]
    training_labels[new_indexes]
    for i_mb in range(n_minibatches):
        if TRAIN_STOP_FLAG:
            break
        a,b = i_mb*minibatch_size, (i_mb+1)*minibatch_size
        iteration = epoch*n_minibatches+i_mb
        images_minibatch = training_images[a:b]
        labels_minibatch = training_labels[a:b]

        asdf = sess.run(training_algorithm,feed_dict={MLP_input: images_minibatch, target: labels_minibatch})

        if iteration % validation_period == 0:
            loss_valid, validation_acc, s = sess.run(
                [loss_function, accuracy, summ],
                feed_dict={MLP_input: validation_images, target: validation_labels})

            writer2.add_summary(s, iteration)

            loss_train, s = sess.run([loss_function, summ],
                                     feed_dict={MLP_input: training_images, target: training_labels})
            writer1.add_summary(s, iteration)

            if loss_valid > prev_validation_loss:
                validation_checks += 1
            else:
                validation_checks = 0
                prev_validation_loss = loss_valid

            print("Epoch: %d/%d, iter: %d. " % (
                epoch+1,
                max_epochs,
                iteration), end='')
            print("Loss (train/val): %.3f / %.3f. Val. accuracy: %.1f%%, Val. checks: %d/%d" %(
                      loss_train,
                      loss_valid,
                      validation_acc*100,
                      validation_checks,
                      max_validation_checks))

            if validation_checks >= max_validation_checks:
                print('Early stopping')
                TRAIN_STOP_FLAG = True
                break
        writer3.add_summary(asdf,i_mb)
writer1.flush()
writer2.flush()
writer3.flush()
