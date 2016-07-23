import tensorflow as tf
import numpy as np

print("-----start-----")

char_rdic = [chr(ch) for ch in range(ord('a'), ord('z')+1)]    # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)}  # char -> id

strings = ["helloworld", "machinegun", "tensorflow", "kwontaeguk",
           "pythongood", "ziptargzbz", "charmcharm", "yohiphoper",
           "soganguniv", "lifeisgood"]
samples = [[char_dic[ch] for ch in string] for string in strings]

"""
x_data = np.array([ [1,0,0,0,0,0,0],  # h
                    [0,1,0,0,0,0,0],  # e
                    [0,0,1,0,0,0,0],  # l
                    [0,0,1,0,0,0,0],  # l
                    [0,0,0,1,0,0,0],  # o
                    [0,0,0,0,1,0,0],  # w
                    [0,0,0,1,0,0,0],  # o
                    [0,0,0,0,0,1,0],  # r
                    [0,0,1,0,0,0,0]],  # l
                     dtype='f')
"""
x_data = [tf.one_hot(sample[:-1], len(char_dic), 1.0, 0.0, -1) for sample in samples]


# Configuration
rnn_size = len(char_dic)
rnn_depth = 3
time_step_size = len(samples[0])-1  # 'helloworl' -> predict 'elloworld'
batch_size = len(strings)

# RNN model
one_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
rnn_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell] * rnn_depth)
state = tf.zeros([batch_size, rnn_cell.state_size])
print(x_data)
X_split = tf.split(1, time_step_size, x_data)
print(X_split)
X_split = [tf.reshape(x, shape=[batch_size, len(char_dic)]) for x in X_split]
print(X_split)

print("..................")

outputs, state = tf.nn.rnn(rnn_cell, X_split, state)
print (state)
print (outputs)

print("----------------")

# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
# targets: list of 1D batch-sized int32 Tensors of the same length as logits.
# weights: list of 1D batch-sized float-Tensors of the same length as logits.
logits = outputs
targets = np.transpose([sample[1:] for sample in samples])
targets = [tf.reshape(target, [-1]) for target in targets]
weights = [tf.ones(shape=[batch_size]) for _ in range(time_step_size)]

print(logits)
print(targets)
print(weights)

loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights)
cost = tf.reduce_mean(tf.reduce_sum(loss))
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

print("======== T R A I N ========")

TRAINING_EPOCHS = 1000
DISPLAY_STEP = 20

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for epoch in range(1, TRAINING_EPOCHS+1):
        sess.run(train_op)
        results = sess.run(tf.arg_max(logits, 2))
        results = np.transpose(results)
        if epoch % DISPLAY_STEP == 1 or epoch == TRAINING_EPOCHS:
            print("Epoch", epoch, "----")
            print("\tCost : {0}".format(sess.run(cost)))
            for result, string in zip(results, strings):
                pred = "".join([char_rdic[t] for t in result])
                real = string[1:]
                if pred == real:
                    res = "PASS"
                else:
                    res = "FAIL"
                print("\tPredict : {0}, Real : {1}, Result : {2}".format(pred, real, res))