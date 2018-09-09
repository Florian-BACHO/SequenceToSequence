import tensorflow as tf

class Encoder:
    def __init__(self, entry, nb_cell):
        self.entry = entry
        batch_size = tf.shape(entry)[1]
        lstm_cell = tf.contrib.rnn.LSTMCell(nb_cell)
        initial_state = lstm_cell.zero_state(batch_size, tf.float32)
        _, self.last_state = tf.nn.dynamic_rnn(lstm_cell, entry, initial_state=initial_state, \
                                               time_major=True)

    # Activate encoder with given batch and return the sequence embedding
    def get_sequence_embedding(self, sess, batch):
        feed_dict = {self.entry: batch}
        return sess.run(self.last_state, feed_dict=feed_dict)
