import tensorflow as tf
import numpy as np

xy_data = np.loadtxt('data/train_data.txt', unpack=True, dtype='float32')
x_data = xy_data[0:-1]
y_data = xy_data[-1]

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

h = tf.matmul(W , X)
# using sigmoid
hypothesis = tf.div(1.0, 1 + tf.exp(-h))

# cost function for logistic classification
cost = tf.reduce_mean((Y - 1) * tf.log(1 - hypothesis) - Y * tf.log(hypothesis))

learning_rate = tf.Variable(0.1)
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
test_x_data = test_xy_data[0:-1]
test_y_data = test_xy_data[-1]

result = sess.run(hypothesis, feed_dict={X : test_x_data})[0]

for x, p_y, y in zip(np.transpose(test_x_data), result, test_y_data):
    prediction = 0
    if p_y > 0.5: prediction = 1
    print("When X = {0} : result = {1}, prediction = {2}, real = {3}".
          format(x, p_y, prediction, y))