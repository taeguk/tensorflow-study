import tensorflow as tf
import input_data

VERSION = "simple.rnn.56.14.2"

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
train_x_data, train_y_data = mnist.train.images, mnist.train.labels
train_data_len = len(train_x_data)
test_x_data, test_y_data = mnist.test.images, mnist.test.labels
test_data_len = len(test_x_data)

RNN_SIZE = 56
RNN_DEPTH = 2
TIMESTEP_SIZE = 14
BATCH_SIZE = 2048

X = tf.placeholder("float", [BATCH_SIZE, 784])
Y = tf.placeholder("float", [BATCH_SIZE, 10])

# RNN model
one_cell = tf.nn.rnn_cell.BasicRNNCell(RNN_SIZE)
rnn_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell] * RNN_DEPTH)
state = tf.zeros([BATCH_SIZE, rnn_cell.state_size])
X_split = tf.split(1, TIMESTEP_SIZE, X)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

initializer = tf.contrib.layers.xavier_initializer()

W = tf.get_variable("W", shape=[RNN_SIZE, 10], initializer=initializer)
b = tf.Variable(tf.random_uniform([10], -1.0, 1.0))

model = tf.matmul(outputs[-1], W) + b
model_with_softmax = tf.nn.softmax(model)

# cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

LEARNING_RATE = 0.001
learning_rate = tf.Variable(LEARNING_RATE)
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    """
        Variables and functions about
        Loading and Saving Data.
    """
    saver = tf.train.Saver()
    SAVE_DIR = 'save_files'
    import os
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    MODEL_SAVE_PATH = "{0}/{1}.{2}.ckpt".format(SAVE_DIR, os.path.basename(__file__), VERSION)
    INFO_FILE_PATH = "{0}/{1}.{2}.info".format(SAVE_DIR, os.path.basename(__file__), VERSION)

    def do_load():
        start_epoch = 1
        try:
            epochs = []
            avg_costs = []
            avg_accuracys = []
            learning_rates = []

            with open(INFO_FILE_PATH, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    data = line.split()
                    epochs.append(int(data[0]))
                    avg_costs.append(float(data[1]))
                    avg_accuracys.append(float(data[2]))
                    learning_rates.append(float(data[3]))
            saver.restore(sess, MODEL_SAVE_PATH)
            print("[*] The save file exists!")

            print("Do you wanna continue? (y/n) ", end="", flush=True)
            if input() == 'n':
                print("not continue...")
                print("[*] Start a training from the beginning.")
                os.remove(INFO_FILE_PATH)
                os.remove(MODEL_SAVE_PATH)
                sess.run(init)
            else:
                print("continue...")
                print("[*] Start a training from the save file.")
                start_epoch = epochs[-1] + 1
                for epoch, avg_cost, avg_accuracy, learning_rate in zip(epochs, avg_costs, avg_accuracys,
                                                                        learning_rates):
                    print("Epoch {0} with learning rate = {1} : avg_cost = {2}, avg_accuracy = {3}".
                          format(epoch, learning_rate, avg_cost, avg_accuracy))

        except FileNotFoundError:
            print("[*] There is no save files.")
            print("[*] Start a training from the beginning.")
            sess.run(init)

        return start_epoch

    def do_save():
        print("[progress] Saving result! \"Never\" exit!!", end='', flush=True)
        saver.save(sess, MODEL_SAVE_PATH)
        with open(INFO_FILE_PATH, "a") as f:
            f.write("{0} {1} {2} {3}".format(epoch, avg_cost, avg_accuracy, LEARNING_RATE) + os.linesep)
        print("", end='\r', flush=True)


    """
        Variables and functions about
        Training and Testing Model
    """
    DISPLAY_SAVE_STEP = 10
    TRAINING_EPOCHS = 500

    def do_train():
        print("[progress] Training model for optimizing cost!", end='', flush=True)
        # Loop all batches for training
        avg_cost = 0
        for end in range(BATCH_SIZE, test_data_len + 1, BATCH_SIZE):
            start = end - BATCH_SIZE
            batch_x = train_x_data[start:end]
            batch_y = train_y_data[start:end]
            data = {X: batch_x, Y: batch_y}
            sess.run(train, feed_dict=data)
            avg_cost += sess.run(cost, feed_dict=data) * len(batch_x) / train_data_len

        print("", end='\r', flush=True)
        return avg_cost

    def do_test():
        print("[progress] Testing model for evaluating accuracy!", end='', flush=True)
        correct_prediction = tf.equal(tf.argmax(model_with_softmax, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Loop all batches for test
        avg_accuracy = 0
        for end in range(BATCH_SIZE, test_data_len+1, BATCH_SIZE):
            start = end - BATCH_SIZE
            batch_x = test_x_data[start:end]
            batch_y = test_y_data[start:end]
            data = {X: batch_x, Y: batch_y}
            avg_accuracy += accuracy.eval(data) * len(batch_x) / test_data_len

        print("", end='\r', flush=True)
        return avg_accuracy


    ##### Start of flow

    start_epoch = do_load()

    if start_epoch == 1:
        avg_accuracy = do_test()
        print("After initializing, accuracy = {0}".format(avg_accuracy))

    # Training cycle
    for epoch in range(start_epoch, TRAINING_EPOCHS + 1):

        avg_cost = do_train()

        # Logging the result
        if epoch % DISPLAY_SAVE_STEP == start_epoch % DISPLAY_SAVE_STEP or epoch == TRAINING_EPOCHS:
            avg_accuracy = do_test()
            do_save()

            # Print Result
            print("Epoch {0} : avg_cost = {1}, accuracy = {2}".format(epoch, avg_cost, avg_accuracy))