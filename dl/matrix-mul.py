import timeit

import tensorflow as tf

with tf.device('/cpu:0'):
    n = 10000
    cpu_a = tf.random.normal([1, n])
    cpu_b = tf.random.normal([n, 1])
    print(cpu_a.device, cpu_b.device)


def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c


cpu_time = timeit.timeit(cpu_run, number=1000000)

with tf.device('/gpu:0'):
    n = 10000
    gpu_a = tf.random.normal([1, n])
    gpu_b = tf.random.normal([n, 1])
    print(gpu_a.device, gpu_b.device)


def gpu_run():
    with tf.device('/gpu:0'):
        d = tf.matmul(gpu_a, gpu_b)
    return d


gpu_time = timeit.timeit(gpu_run, number=1000000)


print("cpu time:",cpu_time)
print("gpu time:",gpu_time)
