import numpy as np


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

    
'''
    TO DO DESCRIPTION AND COMMENTS!!!
'''   
def prepare_sequences(source_sentences, target_sentences, source_dict, target_dict, max_length_source, max_length_target):
    
    source_input, target_input, target_output = [], [], []
    
    for i in range(len(source_sentences)):
        
        # Prepare source sentence
        source = list(source_sentences[i])
        source_mapped = source_dict.text_to_indices(source)
        padded_source = pad_sentence(source_mapped, max_length_source)
        
        # Prepare target sentence
        target = list(target_sentences[i])
        target.insert(0, "<START>")
        target.append("<END>")
        target_mapped = target_dict.text_to_indices(target)

        for j in range(1, len(target_mapped)):
            # Split input and output of target sentence
            input_text, output_text = target_mapped[:j], target_mapped[j]
            padded_target = pad_sentence(input_text, max_length_target)
            
            source_input.append(padded_source)
            target_input.append(padded_target)
            target_output.append(output_text)

    return np.array(source_input), np.array(target_input), np.array(target_output) 
    
    