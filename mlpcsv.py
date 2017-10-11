import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/db/Training.csv"

features = tf.placeholder(tf.float32, name='periods')
classtype = tf.placeholder(tf.int32, name='type')
total = tf.reduce_sum(features, name='total')

printerop = tf.Print(total, [classtype, features, total], name='printer')

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            period,clas,subclas = line.strip().split(",")
            # Run the Print ob
            total = sess.run(printerop, feed_dict={features: period, classtype:clas})
            print(classtype, period)
