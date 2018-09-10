import tensorflow as tf
import numpy as np
from Seq2Seq import *

NB_HIDDEN = 10

SENTENCES = ["This cat is cute", \
             "This cat is eating", \
             "My name is Florian"]
EOS_ONE_HOT_IDX = 0

LEARNING_RATE = 2e-2
STOP_TRESHOLD = 1.0
LOG_DIR = "logs/"

def get_max_sentences_len(splited_sentences):
    out = 0
    for it in splited_sentences:
        length = len(it)
        if length > out:
            out = length
    return out

def make_word_to_int(splited_sentences):
    out = {}
    vocab_counter = 1 # EOS
    for it in splited_sentences:
        for word in it:
            if word not in out:
                out[word] = vocab_counter
                vocab_counter += 1
    return out, vocab_counter

# Convert sentenses list to one hot batch
def convert_to_batch(sentences):
    nb_sentence = len(sentences)
    splited_sentences = [it.split() for it in sentences]
    max_sentences_len = get_max_sentences_len(splited_sentences) + 1 # + 1 for EOS
    word2int, vocab_size = make_word_to_int(splited_sentences)

    batch_matrix = np.zeros([nb_sentence, max_sentences_len, vocab_size])

    for sentence_idx, sentence in enumerate(splited_sentences):
        sentence_len = len(sentence)
        for step in range(max_sentences_len):
            if step >= sentence_len:
                batch_matrix[sentence_idx][step][EOS_ONE_HOT_IDX] = 1.0
            else:
                batch_matrix[sentence_idx][step][word2int[sentence[step]]] = 1.0

    return np.array(batch_matrix), vocab_size

def make_distance_matrix(vectors):
    nb_vector = len(vectors)
    out = np.zeros([nb_vector, nb_vector])
    for i, v1 in enumerate(vectors):
        for j, v2 in enumerate(vectors):
            out[i][j] = np.linalg.norm(v1 - v2)

    return out

if __name__ == "__main__":
    batch, vocab_size = convert_to_batch(SENTENCES)

    seq2seq = Seq2Seq(vocab_size, NB_HIDDEN, NB_HIDDEN, LEARNING_RATE)
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    loss_placeholder = tf.placeholder(tf.float32, [])
    loss_summary = tf.summary.scalar("Loss", loss_placeholder)
    accuracy_placeholder = tf.placeholder(tf.float32, [])
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy_placeholder)
    summary_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run([init_global, init_local])

        accuracy = 0.0
        epoch = 0
        while accuracy < STOP_TRESHOLD:
            loss, accuracy = seq2seq.train(sess, batch)
            print("Epoch %d: loss: %f, accuracy: %f" % (epoch, loss, accuracy))

            if epoch % 10 == 0:
                feed_dict = {loss_placeholder: loss, accuracy_placeholder: accuracy}
                loss_sum, accuracy_sum = sess.run([loss_summary, accuracy_summary], feed_dict=feed_dict)
                summary_writer.add_summary(loss_sum, epoch)
                summary_writer.add_summary(accuracy_sum, epoch)

            epoch += 1
        summary_writer.close()

        embedding = seq2seq.get_embedding(sess, batch)
        print("Features:")
        print(np.argmax(batch, axis=2))
        print("Predictions:")
        print(np.argmax(seq2seq.forward(sess, batch), axis=2))
        print("Embedding:")
        print(embedding)
        print("Sentences's distances:")
        print(make_distance_matrix(embedding))
