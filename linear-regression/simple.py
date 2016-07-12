import tensorflow as tf

x_data = [1, 3, 5]
y_data = [1, 3, 5]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = tf.Variable(0.05)
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
test_x_data = [0.5, 1.5, 5.0, 10.3, -12.0, -9.68]
for x in test_x_data:
    print("When X = {0} : prediction = {1}, real = {2}".
          format(x, sess.run(hypothesis, feed_dict={X : x}), x))