

'''
    TO DO DESCRIPTION
'''
def max_length_sentence(dataset):
    return max([len(line) for line in dataset])


'''
    TO DO DESCRIPTION
'''
def pad_sentence(tokenized_sentence, max_length_sentence, padding_value=0):
    
    pad_length = max_length_sentence - len(tokenized_sentence)
    sentence = list(tokenized_sentence)
    
    if pad_length > 0:
        return np.pad(tokenized_sentence, (0, pad_length), mode='constant', constant_values=int(padding_value))
    else:
        return sentence