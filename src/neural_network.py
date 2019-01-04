import tensorflow as tf
import numpy as np


def new_weights(shape, name=None):
    """Create new Tensorflow variable for the weights of network.

    Args:
        shape (array of int): Shape of weights.

    Returns:
        tf.Variable: Random truncated normal weights with stddev.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def new_biases(length, name=None):
    """Create new Tensorflow variable for the bias of network.

    Args:
        shape (array of int): Shape of bias.

    Returns:
        tf.Variable: Constant values.
    """
    return tf.Variable(tf.constant(0.1, shape=[length]), name=name)


def embedding_layer(input_x, vocabulary_size, embedding_size):
    """Create embedding layer, matrix NxF, where N rows and F features. Each row is the vector of one word.

    Args:
        input_x (TF tensor): Tensor input.
        vocabulary_size (int): N rows
        embedding_size (int): F columns

    Returns:
        TF tensor: Embedding lookup tensor.
    """
    init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
    embeddings = tf.Variable(init_embeds)
    layer = tf.nn.embedding_lookup(embeddings, input_x)
    
    return layer


def get_decoder_outputs(sess, dec_output_tensor, input_tsr, output_tsr, dec_tsr, drop_tsr, dropout_prob, batch_size, target_length, hidden_size, source, target_in):
    """For attention mechanism, I need output of decoder at each timesteps, before backpropagating errors.

    Args:
        batch_size (int): Batch size.
        target_length (int): Longest length of target sentence.
        hidden_size (int): LSTM hidden layer size.
        source (array of array of int): Batch of sentences in the source language.
        target_in (array of array of int): Batch of decoder input sentences in target language.

    Returns:
        numpy array: all decoder outputs.
    """
    # First decoder output will be zero state
    # Feature multiply by two because of bidirectional lstm
    decoder_output = np.zeros((batch_size, target_length, 2 * hidden_size))
    for i in range(target_length-1):

        fgg = sess.run(dec_output_tensor, feed_dict={
            input_tsr: source,
            output_tsr: target_in,
            dec_tsr: decoder_output,
            drop_tsr: dropout_prob,
            drop_tsr: dropout_prob,
        })
        decoder_output[:,i+1] = fgg[:,i]
        
    return decoder_output


def create_network(input_sequence, output_sequence, keep_prob, decoder_outputs, source_dict_size, target_dict_size, embedding_size, hidden_units):
    """Create Tensorflow graph of neural network.

    Args:
        input_sequence (TF tensor): Placeholder for English sentence.
        output_sequence (TF tensor): Placeholder for German sentence.
        keep_prob (TF tensor): Placeholder for dropout probability.
        decoder_outputs (TF tensor): Placeholder for all decoder outputs.
        source_dict_size (int): Size of English vocabulary.
        target_dict_size (int): Size of German vocabulary.
        embedding_size (int): Size of embedding layer.
        hidden_units (int): Hidden size of GRU cell.
        
    Returns:
        TF tensor: logits after last layer.
        TF tensor: Actual outputs of decoder.
        TF tensor: Mask of length of each sentence (because I mask loss function for padding => Do not backpropagate if padding).
    """
    
    with tf.variable_scope("encoder") as encoding_scope:

        # Embedding layer => Output shape is [batch_size, timesteps, embedding_size]
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)

        # Encoder GRU cells (two because bidirectional)
        lstm_fw_cell = tf.contrib.rnn.GRUCell(hidden_units)
        lstm_bw_cell = tf.contrib.rnn.GRUCell(hidden_units)
        
        # Dropout on GRU cells
        dropout_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob,
                                                   output_keep_prob=keep_prob)

        dropout_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob,
                                                   output_keep_prob=keep_prob)

        # enc_outputs shape == (batch_size, source_max_length, 2 * hidden_size)
        (outputs_fw, outputs_bw), (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                                                            dropout_fw, 
                                                                            dropout_bw, 
                                                                            encoder_embedding,
                                                                            dtype=tf.float32)
        # Concatenate outputs of the two GRU cells
        enc_outputs = tf.concat((outputs_fw, outputs_bw), 2)
        
        # Concatenate only last output of GRU cells
        enc_last_state = tf.concat((last_state_fw, last_state_bw), 1)

    with tf.variable_scope("decoder") as decoding_scope:

        # To calculate attention, first hidden states of decoder will be zero vectors
        hidden_with_time_axis = tf.expand_dims(decoder_outputs, 2)
        print("Previous decoder outputs:  " + str(hidden_with_time_axis))
        
        # MOST BASIC SCORE => np.dot(ENCODER_OUTPUTS, DECODER_PREVIOUS_OUTPUT)
        # BAHDANAU SCORE => tanh( W1 * enc_output + W2 * decoder_output)
        
        # Expand encoder outputs to be multiplied with each decoder output
        expanded_encoder_outputs = tf.tile(tf.expand_dims(enc_outputs, axis=1),[1,decoder_outputs.get_shape()[1],1,1])

        a = tf.layers.dense(inputs=expanded_encoder_outputs, units=64, activation=None)
        b = tf.layers.dense(inputs=hidden_with_time_axis, units=64, activation=None)
        bahdanau_score = tf.layers.dense(inputs=tf.nn.tanh(a + b), units=1, activation=None)
        print("Bahdanau score:  " + str(bahdanau_score))
        
        # attention_weights shape == (batch_size, decoder_times, source_max_length, 1)
        attention_weights = tf.nn.softmax(bahdanau_score, axis=2)
        print("Attention weights:  " + str(attention_weights))
        
        # context_vector shape after sum == (batch_size, decoder_times, hidden_size)
        context_vector = attention_weights * expanded_encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=2)
        print("Context vector:  " + str(context_vector))
        
        # embedding layer
        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)
        print("Embedding layer:  " + str(decoder_embedding))
        
        # x shape after concatenation == (batch_size, decoder_times, embedding_dim + hidden_size)
        x = tf.concat([context_vector, decoder_embedding], axis=-1)
        print("Decoder input:  " + str(x))
        
        mask_decoder_input = tf.cast(tf.sign(output_sequence), tf.float32)
        sequence_length = tf.cast(tf.reduce_sum(mask_decoder_input, 1), tf.int32)
        
        # Decoder GRU cell
        lstm_cell = tf.nn.rnn_cell.GRUCell(2 * hidden_units)
        dropout_dec = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        dec_outputs, _ = tf.nn.dynamic_rnn(cell=dropout_dec, inputs=x, initial_state=enc_last_state,
                                           sequence_length=sequence_length)
        
    # Fully connected layer
    logits = tf.layers.dense(inputs=dec_outputs, units=target_dict_size, activation=None)
    print("Logits: " + str(logits))

    return logits, dec_outputs, mask_decoder_input
