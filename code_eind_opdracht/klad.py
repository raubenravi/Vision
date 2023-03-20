import tensorflow as tf

input_shape = (1,32, 32,1)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
 1, kernel_size=(30,30) )(x)
print(y.shape)
(4, 26, 26, 2)