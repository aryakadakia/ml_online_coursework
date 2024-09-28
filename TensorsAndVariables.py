import tensorflow as tf
import numpy as np

# print(tf.__version__)

tf.debugging.set_log_device_placement(True)  # gives more info on tensors
print(tf.executing_eagerly())

# tensors usually multidimensional but can be scalar too, like this -
x0 = tf.constant(3)
# print(x0)
# print(x0.numpy())

# can do arithmetic
result0 = x0 + 5
# print(result0)

x1 = tf.constant([1.1, 2.9, 3.1, 3.3])
result1 = x1 + tf.constant(5.0)  # adds scalar value to every element in tensor
# print(result1.numpy())

result2 = tf.add(x1, tf.constant(5.0)) # add function

# tensors are immutable, always creates new tensor after function

# can have multiple layers too
x2 = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])

# can change data type
x3 = tf.cast(x1, tf.float32)
x4 = tf.cast(x2, tf.float32)
# print x3

# multiple elements in two tensors
result3 = tf.multiply(x3, x4)
# print(result3)

# convert to numpy array
arr_x1 = x1.numpy()
# ...and vice versa
arr_x5 = np.array([[10,20], [30, 40], [50, 60]])
x5 = tf.convert_to_tensor(arr_x5)

# can do numpy functions on tensors too but not recommended
np.square(x2)
np.sqrt(x2)

# check if something is a tensor
# print(tf.is_tensor(arr_x5))

# create tensor of zeros of shape specified (3x5)
t0 = tf.zeros([3, 5], tf.int32)
# can do same for ones
t1 = tf.ones([2, 3], tf.int32)

# can also reshape (3x5 to 5x3)
t0_reshaped = tf.reshape(t0, (5, 3))
# print(t0_reshaped)

# variables hold tensors
v1 = tf.Variable([[1, 5, 6], [2, 3, 4]])

# can specify data type
v2 = tf.Variable([[1, 5, 6], [2, 3, 4]], dtype=tf.int32)

# operations work on variables too
tf.add(v1, v2)
tf.convert_to_tensor(v1) # convert variable to tensor
v1.numpy() # convert to numpy

v1.assign([[10, 58, 68], [21, 32, 43]]) # assign new values to variable
v1[0, 0].assign(100) # change only one particular value at a certain point
print(v1)

v1.assign_add([[1, 1, 1], [2, 2, 2]]) # adds this to each value
v1.assign_sub([[1, 1, 1], [2, 2, 2]]) # subtracts this to each value

var_a = tf.Variable([2.0, 3.0])
var_b = tf.Variable(var_a) # assigning existing variable in new variable. Updating one would not affect the other

