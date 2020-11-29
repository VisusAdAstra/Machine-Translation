# ## GRU implementation



import tensorflow as tf
import tensorflow.keras.backend as bk
from tensorflow.python.keras.metrics import MeanMetricWrapper


def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  #tf.compat.v1.keras.layers.CuDNNGRU
  #tf.keras.layers.CuDNNGRU
  # if tf.test.is_built_with_cuda():
  #   print("is_built_with_cuda")

  if tf.config.list_physical_devices('GPU'):
    print("GPU")
    return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
  else:
    print("No GPU")
    return tf.keras.layers.GRU(units, 
                                return_sequences=True, 
                                return_state=True, 
                                reset_after=True,
                                recurrent_activation='sigmoid', 
                                recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(enc_units, 
                                return_sequences=True, 
                                return_state=True, 
                                recurrent_activation='sigmoid', 
                                recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


class MaskedCategoricalAccuracy(MeanMetricWrapper):
  def __init__(self, mask_id, name='masked_categorical_accuracy', dtype=None):
      super(MaskedCategoricalAccuracy, self).__init__(
          masked_categorical_accuracy, name, dtype=dtype, mask_id=mask_id)


def masked_categorical_accuracy(y_true, y_pred, mask_id):
    true_ids = bk.argmax(y_true, axis=-1)
    pred_ids = bk.argmax(y_pred, axis=-1)
    maskBool = bk.not_equal(true_ids, mask_id)
    maskInt64 = bk.cast(maskBool, 'int64')
    maskFloatX = bk.cast(maskBool, bk.floatx())

    count = bk.sum(maskFloatX)
    equals = bk.equal(true_ids * maskInt64,
                    pred_ids * maskInt64)
    sum = bk.sum(bk.cast(equals, bk.floatx()) * maskFloatX)
    return sum / count


class ExactMatchedAccuracy(MeanMetricWrapper):
  def __init__(self, mask_id, name='exact_matched_accuracy', dtype=None):
      super(ExactMatchedAccuracy, self).__init__(
          exact_matched_accuracy, name, dtype=dtype, mask_id=mask_id)


def exact_matched_accuracy(y_true, y_pred, mask_id):
    true_ids = bk.argmax(y_true, axis=-1)
    pred_ids = bk.argmax(y_pred, axis=-1)

    maskBool = bk.not_equal(true_ids, mask_id)
    maskInt64 = bk.cast(maskBool, 'int64')

    diff = (true_ids - pred_ids) * maskInt64
    matches = bk.cast(bk.not_equal(diff, bk.zeros_like(diff)), 'int64')
    matches = bk.sum(matches, axis=-1)
    matches = bk.cast(bk.equal(matches, bk.zeros_like(matches)), bk.floatx())
    return bk.mean(matches)

