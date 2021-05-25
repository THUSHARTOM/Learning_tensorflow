import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


#initialising tensors
x = tf.constant(4, shape=[1,1], dtype=tf.float32)
x = tf.constant([[1,2,3],[4,5,6]])

x = tf.ones((3,3))
x = tf.eye(3)
x = tf.random.normal((3,3), mean=0, stddev=1)
x = tf.random.uniform((3,3), minval=0, maxval=1)
x = tf.range(9)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float64)
# print(x)

# Mathematical operations
x = tf.constant([1,2,3])
y = tf.constant([10,10,10])
z = tf.add(x,y)  # x+y also will work perfectly
z = tf.subtract(x,y)
z = tf.divide(x,y)
z = tf.multiply(x,y)
z = tf.tensordot(x,y, axes=1)
z = tf.reduce_sum(x*y, axis=0)

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x,y) # z = x@y also works
# print(z)

# Indexing

x = tf.constant([0,1,2,3,4,5,6])
index = tf.constant([0,5])
x_ind = tf.gather(x, index)
print(x_ind)