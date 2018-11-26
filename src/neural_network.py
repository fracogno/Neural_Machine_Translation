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



def create_networksddddd(input_sequence, output_sequence, input_keep_prob, output_keep_prob, source_dict_size, target_dict_size, embedding_size, hidden_units, number_layers, verbose=0):
    
    with tf.variable_scope("encoder") as encoding_scope:

        # Embedding layer => Output shape is [batch_size, timesteps, embedding_size]
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)
        
        gru_fw_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        gru_bw_cell = tf.nn.rnn_cell.GRUCell(hidden_units)

        
        dropout_fw = tf.contrib.rnn.DropoutWrapper(gru_fw_cell, input_keep_prob=input_keep_prob,
                                                   output_keep_prob=output_keep_prob)

        dropout_bw = tf.contrib.rnn.DropoutWrapper(gru_bw_cell, input_keep_prob=input_keep_prob,
                                                   output_keep_prob=output_keep_prob)


        (outputs_fw, outputs_bw), (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                                                            dropout_fw, 
                                                                            dropout_bw, 
                                                                            encoder_embedding,
                                                                            dtype=tf.float32)
        
        enc_outputs = tf.concat((outputs_fw, outputs_bw), 2)
        last_state = tf.concat((last_state_fw, last_state_bw), 1)
        
        print(last_state_fw)
        print(last_state_bw)
        print(last_state)
        print(enc_outputs)
        print("END ENCODER")
        #lstm_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        #enc_outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, encoder_embedding, dtype=tf.float32)
        
        
    with tf.variable_scope("decoder") as decoding_scope:

        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(last_state, 1)
        print(hidden_with_time_axis)
        
        # score shape == (batch_size, max_length, hidden_size)
        # tanh( W1 * enc_output + W2 * enc_hidden)
        bahdanau_score = tf.nn.tanh(tf.layers.dense(inputs=enc_outputs, units=2 * hidden_units, activation=None) + 
                                    tf.layers.dense(inputs=hidden_with_time_axis, units=2 * hidden_units, activation=None))
        print(bahdanau_score)
        
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to FC layer
        attention_weights = tf.nn.softmax(tf.layers.dense(inputs=bahdanau_score, units=1, activation=None), axis=1)
        print(attention_weights)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        print(context_vector)
        
        # embedding layer
        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)
        print(decoder_embedding)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), decoder_embedding], axis=-1)
        print(x)
        
        gru_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        dec_outputs, dec_last_state = tf.nn.dynamic_rnn(cell=gru_cell, inputs=x, dtype=tf.float32)#, initial_state=last_state)
        #dec_outputs, dec_last_state = tf.nn.dynamic_rnn(cell=gru_cell, inputs=x, initial_state=last_state)
        print(dec_outputs)

    # Fully connected
    logits = tf.layers.dense(inputs=dec_outputs, units=target_dict_size, activation=None)
    print(logits)
    # Print shapes
    if verbose > 0: 
        print(outputs)
        print(last_state)
        print(dec_outputs)
        print(dec_last_state)
        print("Logits: " + str(logits.get_shape().as_list()))
        
    return logits



def create_network_ATTENTION_OK(input_sequence, output_sequence, input_keep_prob, output_keep_prob, source_dict_size, target_dict_size, embedding_size, hidden_units, number_layers, verbose=0):
    
    with tf.variable_scope("encoder") as encoding_scope:

        # Embedding layer => Output shape is [batch_size, timesteps, embedding_size]
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)

        lstm_cell = tf.nn.rnn_cell.GRUCell(hidden_units)

        enc_outputs, last_state = tf.nn.dynamic_rnn(lstm_cell,
                                                encoder_embedding,
                                                dtype=tf.float32)
        
    with tf.variable_scope("decoder") as decoding_scope:

        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(last_state, 1)
        
        # tanh( W1 * enc_output + W2 * enc_hidden)
        bahdanau_score = tf.nn.tanh(tf.layers.dense(inputs=enc_outputs, units=hidden_units, activation=None) + 
                                    tf.layers.dense(inputs=hidden_with_time_axis, units=hidden_units, activation=None))
        
        print(bahdanau_score)
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to FC layer
        attention_weights = tf.nn.softmax(tf.layers.dense(inputs=bahdanau_score, units=1, activation=None), axis=1)
        print(attention_weights)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        print(context_vector)
        
        # embedding layer
        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)
        print(decoder_embedding)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), decoder_embedding], axis=-1)
        print(x)
        
        lstm_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        dec_outputs, dec_last_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)#, initial_state=last_state)
        #dec_outputs, dec_last_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, initial_state=last_state)
        print(dec_outputs)

    # Fully connected
    logits = tf.layers.dense(inputs=dec_outputs, units=target_dict_size, activation=None)
    print(logits)
    # Print shapes
    if verbose > 0: 
        print(outputs)
        print(last_state)
        print(dec_outputs)
        print(dec_last_state)
        print("Logits: " + str(logits.get_shape().as_list()))
        
    return logits



'''
    SINGLE LAYER BIDIRECTIONAL ENCODER + UNIDIRECTIONAL DECODER
'''
def create_network(input_sequence, output_sequence, input_keep_prob, output_keep_prob, source_dict_size, target_dict_size, embedding_size, hidden_units, number_layers, verbose=0):
    
    with tf.variable_scope("encoder") as encoding_scope:

        # Embedding layer => Output shape is [batch_size, timesteps, embedding_size]
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)

        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_units, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_units, forget_bias=1.0)

        
        dropout_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=input_keep_prob,
                                                   output_keep_prob=output_keep_prob)

        dropout_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=input_keep_prob,
                                                   output_keep_prob=output_keep_prob)


        (outputs_fw, outputs_bw), (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                                                            dropout_fw, 
                                                                            dropout_bw, 
                                                                            encoder_embedding,
                                                                            dtype=tf.float32)
        
        last_state = tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat((last_state_fw.c , last_state_bw.c), 1),
                                             h=tf.concat((last_state_fw.h , last_state_bw.h), 1))

        
    with tf.variable_scope("decoder") as decoding_scope:

        # Embedding layer for decoder
        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)
    

        lstm_cell = tf.contrib.rnn.LSTMCell(2 * hidden_units)

        dropout_lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                  input_keep_prob=input_keep_prob,
                                                  output_keep_prob=output_keep_prob)
        

        dec_outputs, last_states = tf.nn.dynamic_rnn(cell=dropout_lstm_cell, inputs=decoder_embedding, initial_state=last_state)

    # Fully connected
    logits = tf.layers.dense(inputs=dec_outputs, units=target_dict_size, activation=None)

    # Print shapes
    if verbose > 0: 
        print("Input sequence: " + str(input_sequence.get_shape().as_list()))
        print("Encoder embedding: " + str(encoder_embedding.get_shape().as_list()))
        print("Encoder FW last_state: " + str(last_state_fw[0].get_shape().as_list()))
        print("Encoder BW last_state: " + str(last_state_bw[0].get_shape().as_list()))
        print("Decoder output: " + str(dec_outputs.get_shape().as_list()))
        print("Logits: " + str(logits.get_shape().as_list()))
        
    return logits





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

        # Create deep Bidirectional LSTM
        enc_outputs_fw, enc_outputs_bw, enc_last_state_fw, enc_last_state_bw = get_multilayer_bidirectional_lstm(
                                                                                                         encoder_embedding, 
                                                                                                         hidden_units, 
                                                                                                         input_keep_prob, 
                                                                                                         output_keep_prob,
                                                                                                         number_layers)
        # Concat the last states of each LSTM in the multilayer
        last_states = tuple(
              [tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat((enc_last_state_fw[i].c , enc_last_state_bw[i].c), 1),
                                             h=tf.concat((enc_last_state_fw[i].h , enc_last_state_bw[i].h), 1))
              for i in range(number_layers)] )

        
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
    