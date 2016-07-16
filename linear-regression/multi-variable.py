import tensorflow as tf
import numpy as np

x_data = np.transpose([[1, 2], [3, 4], [5, 6]])
y_data = [6, 12, 18]

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.matmul(W , X) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("-- start learning --")
data = {X : x_data, Y : y_data}
for step in range(2001):
    sess.run(train, feed_dict=data)
    if step % 20 == 0:
        print("Step {0} :  cost = {1}, W = {2}, b = {3}".
              format(step, sess.run(cost, feed_dict=data), sess.run(W), sess.run(b)))
print("-- learning finished --")

print("-- start test --")
test_x_data = [[0.5, 1.5], [5.0, -2.1], [-12.0, 3.9]]
test_y_data = [4.5, 1.8, -3.2]
for x, y in zip(test_x_data, test_y_data):
    print("When X = {0} : prediction = {1}, real = {2}".
          format(x, sess.run(hypothesis, feed_dict={X : np.transpose([x])}), y))