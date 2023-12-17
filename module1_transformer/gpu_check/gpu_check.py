import tensorflow as tf

def check_gpu():
    try:
        # Try to create a small random tensor on the GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0]])
        print("GPU is available.")
    except RuntimeError as e:
        print("GPU is not available: ", e)

check_gpu()

