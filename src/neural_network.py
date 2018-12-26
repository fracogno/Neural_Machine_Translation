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
        TF tensor: Embedding lookup tensor
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
        numpy array: all decoder outputs
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


def create_network(input_sequence, output_sequence, keep_prob, decoder_outputs, source_dict_size, target_dict_size, embedding_size, hidden_units, number_layers):
    
    with tf.variable_scope("encoder") as encoding_scope:

        # Embedding layer => Output shape is [batch_size, timesteps, embedding_size]
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)

        lstm_fw_cell = tf.contrib.rnn.GRUCell(hidden_units)
        lstm_bw_cell = tf.contrib.rnn.GRUCell(hidden_units)
        
        dropout_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_prob,
                                                   output_keep_prob=keep_prob)

        dropout_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_prob,
                                                   output_keep_prob=keep_prob)

        # enc_outputs shape == (batch_size, source_max_length, 2 * hidden_size)
        #enc_outputs, enc_last_state = tf.nn.dynamic_rnn(lstm_cell, encoder_embedding, dtype=tf.float32)
        (outputs_fw, outputs_bw), (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                                                            dropout_fw, 
                                                                            dropout_bw, 
                                                                            encoder_embedding,
                                                                            dtype=tf.float32)
        enc_outputs = tf.concat((outputs_fw, outputs_bw), 2)
        print(enc_outputs)
        
        
        enc_last_state = tf.concat((last_state_fw, last_state_bw), 1)
        '''last_state = tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat((last_state_fw.c , last_state_bw.c), 1),
                                             h=tf.concat((last_state_fw.h , last_state_bw.h), 1))'''

    with tf.variable_scope("decoder") as decoding_scope:

        # To calculate attention, first hidden states of decoder will be zero vectors
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(decoder_outputs, 2)
        print(hidden_with_time_axis)
        
        # MOST BASIC SCORE => np.dot(ENCODER_OUTPUTS, DECODER_PREVIOUS_OUTPUT)
        # BAHDANAU SCORE => tanh( W1 * enc_output + W2 * decoder_output)
        # Score shape before last V == (B, T, F) | After V is (B, T, 1)
        expanded_encoder_outputs = tf.tile(tf.expand_dims(enc_outputs, axis=1),[1,decoder_outputs.get_shape()[1],1,1])

        a = tf.layers.dense(inputs=expanded_encoder_outputs, units=64, activation=None)
        b = tf.layers.dense(inputs=hidden_with_time_axis, units=64, activation=None)
        print(a)
        print(b)
        print(tf.nn.tanh(a + b))
        bahdanau_score = tf.layers.dense(inputs=tf.nn.tanh(a + b), units=1, activation=None)
        print(bahdanau_score)
        
        # attention_weights shape == (batch_size, source_max_length, 1)
        attention_weights = tf.nn.softmax(bahdanau_score, axis=2)
        print(attention_weights)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * expanded_encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=2)
        print(context_vector)
        
        # embedding layer
        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)
        print(decoder_embedding)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([context_vector, decoder_embedding], axis=-1)
        print(x)
        
        lstm_cell = tf.nn.rnn_cell.GRUCell(2 * hidden_units)
        dropout_dec = tf.contrib.rnn.DropoutWrapper(lstm_cell, 
                                                    input_keep_prob=keep_prob,
                                                    output_keep_prob=keep_prob)
        dec_outputs, _ = tf.nn.dynamic_rnn(cell=dropout_dec, inputs=x, initial_state=enc_last_state)
        print(dec_outputs)
        
    # Fully connected
    logits = tf.layers.dense(inputs=dec_outputs, units=target_dict_size, activation=None)
    print(logits)

    return logits, dec_outputs



'''
    Create neural network:
        - Encoder => Embeddings + Deep Bidirectional LSTM with Dropout
        - Decoder => Embeddings + Deep Unidirectional LSTM, whose initial state is concat of forward and backward of encoder
        - FC layer
        - Softmax over the target vocabulary
'''
def create_networkMULTI(input_sequence, output_sequence, input_keep_prob, output_keep_prob, source_dict_size, target_dict_size, embedding_size, hidden_units, number_layers, verbose=0):
    
    with tf.variable_scope("encoder") as encoding_scope:

        # Embedding layer => Output shape is [batch_size, timesteps, embedding_size]
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)

        # Where to accumulate cels
        lstm_layers_vector_fw = []
        lstm_layers_vector_bw = []

        for _ in range(number_layers):

            # Forward and backward direction cell
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_units, forget_bias=1.0)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_units, forget_bias=1.0)

            # Dropout to generalize better
            dropout_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=input_keep_prob,
                                                   output_keep_prob=output_keep_prob)

            dropout_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=input_keep_prob,
                                                   output_keep_prob=output_keep_prob)
            # Append layers
            lstm_layers_vector_fw.append(dropout_fw)
            lstm_layers_vector_bw.append(dropout_bw)

        # Multi RNN layer
        multi_fw_cells = tf.contrib.rnn.MultiRNNCell(lstm_layers_vector_fw, state_is_tuple=True)
        multi_bw_cells = tf.contrib.rnn.MultiRNNCell(lstm_layers_vector_bw, state_is_tuple=True)

        # Input shape of any RNN should be [batch_size, embedding_size] and unpack outputs for forward and backward
        (outputs_fw, outputs_bw), (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                                                        multi_fw_cells, 
                                                                        multi_bw_cells, 
                                                                        inputs, 
                                                                        initial_state_fw=initial_state_fw,
                                                                        initial_state_bw=initial_state_bw,
                                                                        dtype=tf.float32)
        
        # Concat the last states of each LSTM in the multilayer
        last_states = tuple(
              [tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat((last_state_fw[i].c , last_state_bw[i].c), 1),
                                             h=tf.concat((last_state_fw[i].h , last_state_bw[i].h), 1))
              for i in range(number_layers)])

        
    with tf.variable_scope("decoder") as decoding_scope:

        # Embedding layer for decoder
        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)
    
        # Create multi layer unidirectional decoder LSTM cells
        lstm_layers_vector = []
        for _ in range(number_layers):
            
            # Multiply hidden_units by two because I take states from bidirectional LSTM encoder
            lstm_cell = tf.contrib.rnn.LSTMCell(2 * hidden_units)

            # Dropout layer
            dropout_lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                  input_keep_prob=input_keep_prob,
                                                  output_keep_prob=output_keep_prob)
            # Append to multilayer
            lstm_layers_vector.append(dropout_lstm_cell)

        # Multi RNN layer
        multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_layers_vector, state_is_tuple=True)

        # Multi layer dynamic RNN
        dec_outputs, last_states = tf.nn.dynamic_rnn(cell=multi_cell, inputs=decoder_embedding, initial_state=last_states)

    # Fully connected
    logits = tf.layers.dense(inputs=dec_outputs, units=target_dict_size, activation=None)

    # Print shapes
    if verbose > 0: 
        print("Input sequence: " + str(input_sequence.get_shape().as_list()))
        print("Encoder embedding: " + str(encoder_embedding.get_shape().as_list()))
        print("Encoder FW last_state: " + str(enc_last_state_fw[0][0].get_shape().as_list()))
        print("Encoder BW last_state: " + str(enc_last_state_bw[0][0].get_shape().as_list()))
        print("Decoder output: " + str(dec_outputs.get_shape().as_list()))
        print("Logits: " + str(logits.get_shape().as_list()))
        
    return logits    
    