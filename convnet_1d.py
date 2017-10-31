from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import json
import numpy as np
import tensorflow as tf
from os.path import join, splitext

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn_stars(features, labels, mode, bz=-1):

  batch_size = bz
  input_layer = tf.reshape(features["x"], [batch_size, 200, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=32s,
      kernel_size=[3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[2], strides=2)

  pool1_flat = tf.reshape(pool1, [-1, 100])

  dense = tf.layers.dense(inputs=pool1_flat, units=100, activation=tf.nn.relu)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=7)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=7)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
  file = open("database.csv", 'r')
  train_data = np.Array()
  train_labels = np.Array()
  for line in file:
      np.append(train_data,json.load(line.split(';')[1]))
      np.append(train_labels,json.load(line.split(';')[2]))
  train_data.reshape(train_data.shape(0)/200,200)
  train_labels.reshape(train_labels.shape(0),1)

  # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # train_data = mnist.train.images  # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn_stars)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=200,
      hooks=[logging_hook])
  print ("algo hice")
  # Evaluate the model and print results
  #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": eval_data},
#      y=eval_labels,
#      num_epochs=1,
#      shuffle=False)
 # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results)


if __name__ == "__main__":
  tf.app.run()
