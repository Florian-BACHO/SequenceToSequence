import tensorflow as tf
import numpy as np
from Seq2Seq import *

NB_HIDDEN = 50
BATCH_SIZE = 100
NB_STEP = 5
NB_FEATURES = 10
LEARNING_RATE = 2e-2
STOP_TRESHOLD = 0.95
LOG_DIR = "logs/"

# Random one hot batch generator tensor
random_one_hot_batch_generator = tf.one_hot(tf.random_uniform([NB_STEP], minval=0, \
                                                              maxval=NB_FEATURES - 1, \
                                                              dtype=tf.int32), NB_FEATURES)

if __name__ == "__main__":
    seq2seq = Seq2Seq(NB_FEATURES, NB_HIDDEN, NB_HIDDEN, LEARNING_RATE)
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run([init_global, init_local])

        accuracy = (0.0, 0.0)
        epoch = 0
        while accuracy[0] < STOP_TRESHOLD:
            epoch += 1

            batch = np.array([sess.run(random_one_hot_batch_generator) for _ in range(BATCH_SIZE)])

            loss, accuracy = seq2seq.train(sess, batch)
            print("Epoch %d: loss: %f, accuracy: %f" % (epoch, loss, accuracy[0]))
