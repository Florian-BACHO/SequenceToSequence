import tensorflow as tf
import numpy as np
import time
from Seq2Seq import *

NB_HIDDEN = 150
BATCH_SIZE = 50
NB_STEP = 10
NB_FEATURES = 26 # Alphabet
LEARNING_RATE = 2e-2
STOP_TRESHOLD = 1.
LOG_DIR = "logs/"

# Random one hot batch generator tensor
random_one_hot_batch_generator = tf.one_hot(tf.random_uniform([NB_STEP], minval=0, \
                                                              maxval=NB_FEATURES - 1, \
                                                              dtype=tf.int32), NB_FEATURES)

if __name__ == "__main__":
    seq2seq = Seq2Seq(NB_FEATURES, NB_HIDDEN, NB_HIDDEN, LEARNING_RATE)
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    loss_placeholder = tf.placeholder(tf.float32, [])
    loss_summary = tf.summary.scalar("Loss", loss_placeholder)
    accuracy_placeholder = tf.placeholder(tf.float32, [])
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy_placeholder)
    summary_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run([init_global, init_local])

        accuracy = (0.0, 0.0)
        epoch = 0
        while accuracy[0] < STOP_TRESHOLD:
            batch = np.array([sess.run(random_one_hot_batch_generator) for _ in range(BATCH_SIZE)])

            loss, accuracy = seq2seq.train(sess, batch)
            print("Epoch %d: loss: %f, accuracy: %f" % (epoch, loss, accuracy[0]))

            if epoch % 10 == 0:
                feed_dict = {loss_placeholder: loss, accuracy_placeholder: accuracy[0]}
                loss_sum, accuracy_sum = sess.run([loss_summary, accuracy_summary], feed_dict=feed_dict)
                summary_writer.add_summary(loss_sum, epoch)
                summary_writer.add_summary(accuracy_sum, epoch)

            epoch += 1
        summary_writer.close()
