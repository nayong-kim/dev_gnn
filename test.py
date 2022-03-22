import tensorflow as tf
import numpy as np



empty = np.array([1, 2, 3])
print(empty)
empty = tf.TensorArray( empty)
print(empty)