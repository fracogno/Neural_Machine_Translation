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
    TODO DESCRIPTION
'''
def create_network(input_sequence, output_sequence, source_dict_size, target_dict_size, embedding_size, lstm_neurons, target_vocabulary_size, verbose=0):

    with tf.variable_scope("encoding") as encoding_scope:
        encoder_embedding = embedding_layer(input_sequence, source_dict_size, embedding_size)
        lstm_enc = tf.nn.rnn_cell.LSTMCell(lstm_neurons)
        enc_outputs, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=encoder_embedding, dtype=tf.float32)


    with tf.variable_scope("decoding") as decoding_scope:
        decoder_embedding = embedding_layer(output_sequence, target_dict_size, embedding_size)
        lstm_dec = tf.nn.rnn_cell.LSTMCell(lstm_neurons)
        dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=decoder_embedding, initial_state=last_state)

        out_shape = dec_outputs.get_shape().as_list()
        dec_outputs = tf.reshape(dec_outputs, [-1, out_shape[1] * out_shape[2]])

    logits = tf.layers.dense(inputs=dec_outputs, units=target_vocabulary_size, activation=None)
    
    # Print shapes
    if verbose > 0: 
        print("Input sequence: " + str(input_sequence.get_shape().as_list()))
        print("Encoder embedding: " + str(encoder_embedding.get_shape().as_list()))
        print("Encoder last_state: " + str(last_state[0].get_shape().as_list()))

        print("Decoder output: " + str(dec_outputs.get_shape().as_list()))
        print("Logits: " + str(logits.get_shape().as_list()))
        
    return logits

    