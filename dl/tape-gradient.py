import tensorflow as tf

a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape:
    tape.watch([w])
    y = a * w ** 2 + b * w + c
    [dy_dw] = tape.gradient(y, [w])
    print(dy_dw)

