import tensorflow as tf

class Decoder:
    def __init__(self, batch_shape, nb_features, initial_state, nb_cell):
        self.batch_size = batch_shape[0]
        self.nb_step = batch_shape[1]
        self.nb_features = nb_features

        self.initial_state = initial_state

        lstm_cell = tf.contrib.rnn.GRUCell(nb_cell)

        lstm_outputs_ta, final_output, last_state = tf.nn.raw_rnn(lstm_cell, self._loop_fn)
        lstm_outputs = tf.transpose(lstm_outputs_ta.stack(), perm=[1, 0, 2])

        outputs_activation = lambda x: tf.layers.dense(x, nb_features)
        self.outputs = tf.map_fn(outputs_activation, lstm_outputs)

    def _loop_fn(self, time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0

        if cell_output is None:
            next_cell_state = self.initial_state
            next_input = tf.zeros([self.batch_size, self.nb_features])
        else:  # pass the last state to the next
            next_cell_state = cell_state
            next_sampled_input = tf.argmax(cell_output, axis=1)
            next_input = tf.one_hot(next_sampled_input, self.nb_features)

        elements_finished = (time >= self.nb_step)

        next_loop_state = None

        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    # Return decoder outputs
    def forward(self, sess, feed_dict):
        return sess.run(self.outputs, feed_dict=feed_dict)
