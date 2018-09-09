import tensorflow as tf
import numpy as np

NB_HIDDEN = 25
BATCH_SIZE = 100
NB_STEP = 15
NB_FEATURES = 10
LEARNING_RATE = 1e-2
STOP_TRESHOLD = 0.95

# Random one hot batch generator tensor
random_one_hot_batch_generator = tf.one_hot(tf.random_uniform([NB_STEP], minval=0, \
                                                              maxval=NB_FEATURES - 1, \
                                                              dtype=tf.int32), NB_FEATURES)

class Seq2Seq:
    def __init__(self, nb_features, nb_encoder_cell, nb_decoder_cell):
        self._create_encoder(nb_features, nb_encoder_cell)
        self._create_decoder(nb_features, nb_decoder_cell)
        self._create_learning_tensors()

    # Encoder
    def _create_encoder(self, nb_features, nb_encoder_cell):
        with tf.variable_scope("encoder"):
            self.features = tf.placeholder(tf.float32, [None, None, nb_features]) # Batch, step, features
            self.features_t = tf.transpose(self.features, perm=[1, 0, 2]) # Step, batch, features
            self.batch_size = tf.shape(self.features)[0]
            self.nb_step = tf.shape(self.features)[1]
            self.nb_features = nb_features

            lstm_cell = tf.contrib.rnn.LSTMCell(nb_encoder_cell)

            initials_states = lstm_cell.zero_state(self.batch_size, tf.float32)

            _, self.encoder_last_states = tf.nn.dynamic_rnn(lstm_cell, self.features_t, initial_state=initials_states, time_major=True)

    # Decoder
    def _create_decoder(self, nb_features, nb_decoder_cell):
        with tf.variable_scope("decoder"):

            lstm_cell = tf.contrib.rnn.LSTMCell(nb_decoder_cell)

            lstm_outputs_ta, final_decoder_output, last_decoder_state = tf.nn.raw_rnn(lstm_cell, self._decoder_loop_fn)
            lstm_outputs = lstm_outputs_ta.stack()

        outputs_activation = lambda x: tf.layers.dense(x, self.nb_features)
        self.decoder_outputs = tf.map_fn(outputs_activation, lstm_outputs)
        self.decoder_outputs_t = tf.transpose(self.decoder_outputs, perm=[1, 0, 2])

    def _create_learning_tensors(self):
        loss = tf.losses.mean_squared_error(self.features_t, self.decoder_outputs)
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.training_op = optimizer.minimize(self.loss)

        self.accuracy = tf.metrics.accuracy(tf.argmax(self.features_t, axis=2), tf.argmax(self.decoder_outputs, axis=2))

    def _decoder_loop_fn(self, time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0

        if cell_output is None:
            next_cell_state = self.encoder_last_states
            next_input = tf.zeros([self.batch_size, self.nb_features])
        else:  # pass the last state to the next
            next_cell_state = cell_state
            next_sampled_input = tf.argmax(cell_output, axis=1)
            next_input = tf.one_hot(next_sampled_input, self.nb_features)

        elements_finished = (time >= self.nb_step)

        next_loop_state = None

        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    # Activate the encoder / decoder with given batch
    def forward(self, session, batch, get_loss=False, get_accuracy=False):
        feed_dict = {self.features: batch}
        tensors = [self.decoder_outputs_t]

        if get_loss:
            tensors.append(self.loss)
        if get_accuracy:
            tensors.append(self.accuracy)

        return session.run(tensors, feed_dict=feed_dict)

    # Execute a train step
    def train(self, session, batch, initial_states=None):
        feed_dict = {self.features: batch}
        loss, accuracy, _ = session.run([self.loss, self.accuracy, self.training_op], feed_dict=feed_dict)

        return loss, accuracy

if __name__ == "__main__":
    seq2seq = Seq2Seq(NB_FEATURES, NB_HIDDEN, NB_HIDDEN)
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
