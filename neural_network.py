import tensorflow as tf


'''
    TODO DESCRIPTION
'''
def new_weights(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


'''
    TODO DESCRIPTION
'''
def new_biases(length, name=None):
    return tf.Variable(tf.constant(0.1, shape=[length]), name=name)


'''
    TODO DESCRIPTION
'''
def embedding_layer(input_x, vocabulary_size, embedding_size):
    init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
    embeddings = tf.Variable(init_embeds)
    layer = tf.nn.embedding_lookup(embeddings, input_x)
    
    return layer


'''
    Create graph for a Deep Bidirectional LSTM
'''
def get_multilayer_bidirectional_lstm(inputs, hidden_units, input_keep_prob, output_keep_prob, number_layers,
                                      initial_state_fw=None, initial_state_bw=None):

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

    return outputs_fw, outputs_bw, last_state_fw, last_state_bw


'''
    TODO DESCRIPTION
'''
def create_network(input_sequence, output_sequence, input_keep_prob, output_keep_prob, source_dict_size, target_dict_size, embedding_size, hidden_units, number_layers, verbose=0):
    
    with tf.variable_scope("encoding") as encoding_scope:

        # Embedding layer => Output shape is [batch_size, timesteps, embedding_size]
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)

        # Create deep Bidirectional LSTM
        enc_outputs_fw, enc_outputs_bw, enc_last_state_fw, enc_last_state_bw = get_multilayer_bidirectional_lstm(
                                                                                                         encoder_embedding, 
                                                                                                         hidden_units, 
                                                                                                         input_keep_prob, 
                                                                                                         output_keep_prob,
                                                                                                         number_layers)
    with tf.variable_scope("decoding") as decoding_scope:

        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)

        dec_outputs_fw, dec_outputs_bw, _, _ = get_multilayer_bidirectional_lstm(decoder_embedding, 
                                                                                 hidden_units, 
                                                                                 input_keep_prob, 
                                                                                 output_keep_prob,
                                                                                 number_layers,
                                                                                 enc_last_state_fw,
                                                                                 enc_last_state_bw)
    # Concat outputs
    dec_outputs_concat = tf.concat([dec_outputs_fw, dec_outputs_bw], 1) 

    # Reshape outputs
    out_shape = dec_outputs_concat.get_shape().as_list()
    dec_outputs = tf.reshape(dec_outputs_concat, [-1, out_shape[1] * out_shape[2]])
    
    # Fully connected
    logits = tf.layers.dense(inputs=dec_outputs, units=target_dict_size, activation=None)

    # Print shapes
    if verbose > 0: 
        print("Input sequence: " + str(input_sequence.get_shape().as_list()))
        print("Encoder embedding: " + str(encoder_embedding.get_shape().as_list()))
        print("Encoder FW last_state: " + str(enc_last_state_fw[0][0].get_shape().as_list()))
        print("Decoder concatenated output: " + str(dec_outputs_concat.get_shape().as_list()))
        print("Logits: " + str(logits.get_shape().as_list()))
        
    return logits    
    