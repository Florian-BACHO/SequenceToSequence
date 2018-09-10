import tensorflow as tf
from Encoder import *
from Decoder import *

class Seq2Seq:
    def __init__(self, nb_features, nb_encoder_cell, nb_decoder_cell, learning_rate=1e-2):
        self.features = tf.placeholder(tf.float32, [None, None, nb_features]) # Batch, step, features

        batch_shape = tf.shape(self.features)
        self.batch_size = batch_shape[0]
        self.nb_step = batch_shape[1]
        self.nb_features = nb_features

        with tf.variable_scope("Encoder"):
            self.encoder = Encoder(self.features, nb_encoder_cell)

        with tf.variable_scope("decoder"):
            self.decoder = Decoder(batch_shape, nb_features, self.encoder.last_state, nb_decoder_cell)

        self._create_learning_tensors(learning_rate)

    def _create_learning_tensors(self, learning_rate):
        loss = tf.losses.mean_squared_error(self.features, self.decoder.outputs)
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = optimizer.minimize(self.loss)

        features_argmax = tf.argmax(self.features, axis=2)
        predi_argmax = tf.argmax(self.decoder.outputs, axis=2)
        equality = tf.equal(predi_argmax, features_argmax)
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    # Activate the encoder / decoder with given batch
    def forward(self, session, batch):
        feed_dict = {self.features: batch}
        return self.decoder.forward(session, feed_dict)

    def get_embedding(self, session, batch):
        return self.encoder.get_sequence_embedding(session, batch)

    # Execute a train step
    def train(self, session, batch, initial_states=None):
        feed_dict = {self.features: batch}
        loss, accuracy, _ = session.run([self.loss, self.accuracy, self.training_op], feed_dict=feed_dict)

        return loss, accuracy
