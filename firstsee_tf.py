# *************************
# @Time    : 2019-11-15 18:21
# @Author  : Qi
# @Site    : 
# @File    : firstsee_tf.py
# @Function: 
# *************************

import tensorflow as tf

tf.enable_eager_execution()
# 张量，形状和类型
a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)
A = tf.constant([[1, 2], [3, 4]])

B = tf.constant([[5, 6], [7, 8]])
# 张量操作
C = tf.matmul(A, B)
print(c)
print(C)

x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))
with tf.GradientTape() as tape:
    y = tf.square(x)

y_grad = tape.gradient(y, x) # y关于x的导数
print([y.numpy(), y_grad.numpy()])


