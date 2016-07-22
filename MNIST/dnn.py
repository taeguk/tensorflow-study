import tensorflow as tf
import numpy as np
import input_data

VERSION = "adam.dropout0.7.L6.0.1"

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

def make_model(X, dropout_rate):
    L = input_layer(X, 784, 256)
    L = hidden_layer(L, 256, 512, dropout_rate)
    L = hidden_layer(L, 512, 32, dropout_rate)
    L = hidden_layer(L, 32, 64, dropout_rate)
    L = hidden_layer(L, 64, 64, dropout_rate)
    return output_layer(L, 64, 10, dropout_rate)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
train_x_data, train_y_data = mnist.train.images, mnist.train.labels
test_x_data, test_y_data = mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

dropout_rate = tf.placeholder("float")

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

model = make_model(X, dropout_rate)
model_with_softmax = tf.nn.softmax(model)

# cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

f_learning_rate = 0.01
learning_rate = tf.Variable(f_learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    # Training cycle
    print("-- start learning --")
    display_save_step = 1
    training_epochs = 1024
    batch_size = 1024
    start_epoch = 1

    saver = tf.train.Saver()
    import os
    save_dir = 'save_files'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model_save_path = "{0}/{1}.{2}.ckpt".format(save_dir, os.path.basename(__file__), VERSION)
    info_file_path = "{0}/{1}.{2}.info".format(save_dir, os.path.basename(__file__), VERSION)
    try:
        epochs = []
        avg_costs = []
        learning_rates = []

        with open(info_file_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data = line.split()
                epochs.append(int(data[0]))
                avg_costs.append(float(data[1]))
                learning_rates.append(float(data[2]))
        saver.restore(sess, model_save_path)
        print("[*] The save file exists!")

        print("Do you wanna continue? (y/n) ", end="", flush=True)
        if input() == 'n':
            print("not continue...")
            print("[*] Start a training from the beginning.")
            os.remove(info_file_path)
            os.remove(model_save_path)
            sess.run(init)
        else:
            print("continue...")
            print("[*] Start a training from the save file.")
            start_epoch = epochs[-1] + 1
            for epoch, avg_cost, learning_rate in zip(epochs, avg_costs, learning_rates):
                print("Epoch {0} with learning rate = {1} : avg_cost = {2}".
                      format(epoch, learning_rate, avg_cost))

    except FileNotFoundError:
        print("[*] There is no save files.")
        print("[*] Start a training from the beginning.")
        sess.run(init)

    for epoch in range(start_epoch, training_epochs+1):
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_cost = 0

        # Loop all batches
        for i in range(batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            data = {X : batch_x, Y : batch_y, dropout_rate : 0.7}
            sess.run(train, feed_dict=data)
            avg_cost += sess.run(cost, feed_dict=data) / total_batch

        if epoch % display_save_step == 0 or epoch == training_epochs:
            print("[*] Result is saving! \"Never\" exit!!", end='\r')
            saver.save(sess, model_save_path)
            with open(info_file_path, "a") as f:
                f.write("{0} {1} {2}".format(epoch, avg_cost, f_learning_rate) + os.linesep)
            print("                                          ", end='\r')
            print("Epoch {0} : avg_cost = {1}".format(epoch, avg_cost))

    print("-- learning finished --")

    print("-- start test --")
    correct_prediction = tf.equal(tf.argmax(model_with_softmax, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy : {0}".format(accuracy.eval({X : mnist.test.images, Y : mnist.test.labels, dropout_rate : 1.0})))