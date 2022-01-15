"""
  Trains the co-occurrence matrix.
  See the GloVe paper for the math.
  https://nlp.stanford.edu/pubs/glove.pdf
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from cooccurrence_matrix import CooccurrenceGenerator

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input cooccur.pb.b64.bz2 file.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("embedding_dim", 64,
                     "Embedding dimension.")
flags.DEFINE_integer("batch_size", 1024,
                     "Batch size")
flags.DEFINE_integer("shuffle_buffer_size", 5000000,
                     "Shuffle buffer size")
flags.DEFINE_string("terms", None, "CSV of terms to dump")
flags.DEFINE_string("tensorboard_dir", None, "Location to store training logs.")
flags.DEFINE_string("checkpoint_dir", None, "Location to save checkpoints.")
flags.DEFINE_integer("steps_per_epoch", 1000,
                     "Number of steps per epoch")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of epochs")
flags.DEFINE_integer("validation_steps", 100,
                     "Number of validation steps")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")

# Required flag.
flags.mark_flag_as_required("input_file")


class NNCallback(Callback):
    """Nearest neighbor callback."""
    def __init__(self, csv, token_dictionary):
        super(NNCallback, self).__init__()
        tokens = csv.split(',')
        self.tokens = []
        self.indices = []
        self.num_tokens = token_dictionary.get_dictionary_size()
        for token in tokens:
            index = token_dictionary.get_token_index(token)
            if index is not None:
                self.tokens.append(token)
                self.indices.append(index)
        self.indices = np.asarray(self.indices, dtype=np.int32)
        self.indices_as_tensors = tf.convert_to_tensor(self.indices)
        self.token_dictionary = token_dictionary

    def on_epoch_end(self, epoch, logs=None):
        embedding_model = Model(inputs=self.model.get_layer("token").input,
                                outputs=self.model.get_layer("word_embedding").output)
        all_indices = np.array(np.arange(self.num_tokens), dtype=np.int32)
        all_indices = tf.convert_to_tensor(all_indices)
        embeddings = embedding_model(all_indices)
        target_embeddings = embedding_model(self.indices_as_tensors)
        # Find distances from all target embeddings to all other embeddings.
        results = K.dot(target_embeddings, K.transpose(embeddings))
        results = K.get_session().run(results)
        count = min(FLAGS.max_terms, self.num_tokens)
        for i in range(len(self.tokens)):
            far_to_near_indices = np.argsort(results[i])
            result_list = []
            for j in range(count):
                idx = far_to_near_indices[self.num_tokens - 1 - j]
                sim = results[i][idx]
                other_token = self.token_dictionary.get_token(idx)
                display = '%s:%3f' % (other_token, sim)
                result_list.append(display)
            print('Nearest to %s: %s' % (self.tokens[i], ','.join(result_list)))


def make_callbacks(token_dictionary):
    """Makes the model hook callbacks."""
    callbacks = []
    if FLAGS.tensorboard_dir is not None:
        callback = TensorBoard(log_dir=FLAGS.tensorboard_dir,
                               histogram_freq=100,
                               batch_size=FLAGS.batch_size,
                               write_graph=True,
                               update_freq='epoch')
        callbacks.append(callback)
    if FLAGS.terms is not None:
        callback = NNCallback(FLAGS.terms, token_dictionary)
        callbacks.append(callback)
    if FLAGS.checkpoint_dir is not None:
        checkpointer = ModelCheckpoint(filepath=FLAGS.checkpoint_dir,
                                       verbose=1, mode='min')
        callbacks.append(checkpointer)
    if len(callbacks) == 0:
        return None
    return callbacks


# Define glove loss.
def glove_loss(y_true, y_pred):
    """The GloVe weighted loss."""
    weight = K.minimum(1.0, y_true / 100.0)
    weight = K.pow(weight, 0.75)
    # Approximate log base 10.
    log_true = K.log(1.0 + y_true) / 2.3
    return K.mean(K.square(log_true - y_pred) * weight, axis=-1)


def main(argv):
    """Main function."""
    del argv  # Unused.
    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    num_tokens = token_dictionary.get_dictionary_size()

    # First token
    word_embedding = Embedding(output_dim=FLAGS.embedding_dim,
                               input_dim=num_tokens,
                               input_length=1,
                               embeddings_initializer='he_normal',
                               name='word_embedding')

    token1 = Input(shape=(1,), dtype='int32', name='token')
    token2 = Input(shape=(1,), dtype='int32', name='other_token')

    bias = Embedding(output_dim=1,
                     input_dim=num_tokens,
                     input_length=1,
                     embeddings_initializer='zeros')
    embed1 = word_embedding(token1)
    bias1 = bias(token1)
    embed2 = word_embedding(token2)
    bias2 = bias(token2)
    dot = Dot(axes=2)([embed1, embed2])
    output = Add()([dot, bias1, bias2])
    output = Flatten()(output)

    model = Model(inputs=[token1, token2], outputs=output)
    optimizer = Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=optimizer, loss=glove_loss)

    train_data = CooccurrenceGenerator(FLAGS.input_file)
    validation_data = CooccurrenceGenerator(FLAGS.input_file)

    train_iterator = train_data.get_batch(FLAGS.batch_size, FLAGS.shuffle_buffer_size)
    validation_iterator = validation_data.get_batch(FLAGS.batch_size, FLAGS.shuffle_buffer_size)

    callbacks = make_callbacks(token_dictionary)

    model.fit_generator(train_iterator,
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        epochs=FLAGS.num_epochs,
                        callbacks=callbacks,
                        validation_data=validation_iterator,
                        validation_steps=FLAGS.validation_steps)

if __name__ == "__main__":
    app.run(main)
