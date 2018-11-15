
'''
    TO DO DESCRIPTION
'''
class LanguageDictionary:
    
    special_words = ["<PAD>", "<START>", "<END>", "<UNK>"]
    
    def __init__(self, sentences):
        
        # I want to have a unique mapping between a word and a corresponding integer (and vice versa)
        self.index_to_word = list()
        self.word_to_index = dict()
        
        # Mapping for special words needed
        for i in range(0,len(self.special_words)):
            self.word_to_index[self.special_words[i]] = i
            self.index_to_word.append(self.special_words[i])
            
        # Iterate over the sentences in the training data
        current_index = len(self.special_words)
        for sample in sentences:
            
            # Iterate over each word in a sentence
            for word in sample:
    
                # Add word if not present in the dictionary
                if word not in self.word_to_index:
                    self.word_to_index[word] = current_index
                    self.index_to_word.append(word)
                    current_index += 1
                   
        # Assert that same number of words in the mapping
        assert(len(self.word_to_index.keys()) == len(self.index_to_word))
        

    def text_to_indices(self, text_tokens):
        mapped_sentence = list()
        
        # Convert each token word into its corresponding number
        for word in text_tokens:
            
            # If word is present in the dictionary, append the corresponding index
            if word in self.word_to_index:
                mapped_sentence.append(self.word_to_index[word])
                
            else: # Otherwise append UNKNOWN WORD INDEX
                mapped_sentence.append(self.word_to_index["<UNK>"])
        
        return mapped_sentence


    def indices_to_text(self, indices_array):
        mapped_text = list()
        
        # Iterate over array of indices
        for i in indices_array:
            mapped_text.append(self.index_to_word[i])
        
        return " ".join(mapped_text)

