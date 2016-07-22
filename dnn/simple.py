import tensorflow as tf
import numpy as np

initializer = tf.contrib.layers.xavier_initializer()

def input_layer(X, num_input, num_output):
    global layer_cnt
    layer_cnt = 1
    W = tf.get_variable("W" + str(layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))

def hidden_layer(X, num_input, num_output, dropout_rate):
    global layer_cnt
    layer_cnt += 1
    X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("W" + str(layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))

def output_layer(X, num_input, num_output, dropout_rate):
    global layer_cnt
    layer_cnt += 1
    X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("W" + str(layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(tf.random_uniform([num_output], -1.0, 1.0))
    return tf.add(tf.matmul(X, W), b)

train_xy_data = np.loadtxt('data/train_data.txt', unpack=True, dtype='float32')
train_x_data = np.transpose(train_xy_data[0:2])
train_y_data = np.transpose(train_xy_data[2:])

test_xy_data = np.loadtxt('data/test_data.txt', unpack=True, dtype='float32')
test_x_data = np.transpose(test_xy_data[0:2])
test_y_data = np.transpose(test_xy_data[2:])

# (the number of data) x (x1 x2)
X = tf.placeholder(tf.float32, [None, 2])
# (the number of data) x (y1 y2 y3)
Y = tf.placeholder(tf.float32, [None, 3])

dropout_rate = tf.placeholder(tf.float32)

L1 = input_layer(X, 2, 16)
L2 = hidden_layer(L1, 16, 256, dropout_rate)
L3 = hidden_layer(L2, 256, 1024, dropout_rate)
L4 = hidden_layer(L3, 1024, 768, dropout_rate)
L5 = hidden_layer(L4, 768, 128, dropout_rate)
model = L6 = output_layer(L5, 128, 3, dropout_rate)

# cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

learning_rate = tf.Variable(0.001)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    print("-- start learning --")
    data = {X : train_x_data, Y : train_y_data, dropout_rate : 1}
    training_epochs = 200
    for epoch in range(training_epochs):

        sess.run(train, feed_dict=data)
        print("Epoch {0} :  cost = {1}".
                format(epoch+1, sess.run(cost, feed_dict=data)))
    print("-- learning finished --")

    print("-- start test --")
    hypothesis = tf.nn.softmax(model)
    result = sess.run(hypothesis, feed_dict={X : test_x_data, dropout_rate : 1})

    for x, p_y, y in zip(test_x_data, result, test_y_data):
        prediction = sess.run(tf.arg_max(p_y, 0))
        real = sess.run(tf.arg_max(y, 0))
        print("When X = {0} : result = {1}, prediction = {2}, real = {3}".
              format(x, p_y, prediction, real))