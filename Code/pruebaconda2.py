import sys
import tensorflow as tf


def print_tensor_info(tensor):
    print("Tensor shape: ", tensor.shape)
    print("Tensor data type: ", tensor.dtype)
    print("Tensor values: ", tensor.numpy())


#print("Python version:", sys.version)

# Create a rank-0 tensor (scalar)
rank_0_tensor = tf.constant(4)
print_tensor_info(rank_0_tensor)

# Create a rank-1 tensor (vector)
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print_tensor_info(rank_1_tensor)

# Create a rank-2 tensor (matrix)
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
print_tensor_info(rank_2_tensor)
