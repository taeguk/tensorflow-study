import tensorflow as tf
import numpy as np

xy_data = np.loadtxt('data/train_data.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy_data[0:3])
y_data = np.transpose(xy_data[3:])

# (b w1 w2) x (0 1 2)
W = tf.Variable(tf.random_uniform([3, 3], -1.0, 1.0))

# (the number of data) x (1 x1 x2)
X = tf.placeholder(tf.float32, [None, 3])
# (the number of data) x (y1 y2 y3)
Y = tf.placeholder(tf.float32, [None, 3])

# using softmax
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

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
        print("Step {0} :  cost = {1}, W = {2}".
              format(step, sess.run(cost, feed_dict=data), sess.run(W)))
print("-- learning finished --")

print("-- start test --")
test_xy_data = np.loadtxt('data/test_data.txt', unpack=True, dtype='float32')
test_x_data = np.transpose(test_xy_data[0:3])
test_y_data = np.transpose(test_xy_data[3:])

result = sess.run(hypothesis, feed_dict={X : test_x_data})

for x, p_y, y in zip(test_x_data, result, test_y_data):
    prediction = sess.run(tf.arg_max(p_y, 0))
    real = sess.run(tf.arg_max(y, 0))
    print("When X = {0} : result = {1}, prediction = {2}, real = {3}".
          format(x, p_y, prediction, real))