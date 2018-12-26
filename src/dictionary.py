
class LanguageDictionary:
    """Add padding to sentences, add <START> and <END> token to target sentence and convert each token to the index of the corresponding vocabulary.

    Args:
        source_sentences (array of array of str): Tokenized sentences in English.
        target_sentences (array of array of str): Tokenized sentences in German.
        source_dict (obj): Dictionary object of source language (English).
        target_dict (obj): Dictionary object of target language (German).
        
    Returns:
        array of array of int: English mapped tokens.
        array of array of int: German mapped tokens.
    """
    # Special words needed for beginning and end sentence, padding and unknown words
    special_words = ["<PAD>", "<START>", "<END>", "<UNK>"]
    
    def __init__(self, sentences, max_length_sentence):
        """Constructor of Dictionary object, 

        Args:
            self (obj): Python reference to current object.
            sentences (array of array of str): All tokenized sentences.
            max_length_sentence (int): Length of longest sentence.
        """
        # I need a unique mapping between a word and a corresponding integer (and vice versa)
        self.index_to_word = list()
        self.word_to_index = dict()
        self.max_length_sentence = max_length_sentence
        
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
        """Transform tokenized sentence of word to the corresponding tokenized sentence of integers.

        Args:
            self (obj): Python reference to current object.
            text_tokens (array of str): Tokenized sentence of words.

        Returns:
            array of int: Tokenized sentence of integers.
        """
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
        """Transform tokenized sentence of integers to the corresponding tokenized sentence of words.

        Args:
            self (obj): Python reference to current object.
            indices_array (array of int): Tokenized sentence of integers.

        Returns:
            array of str: Tokenized sentence of words.
        """
        mapped_text = list()
        
        # Iterate over array of indices
        for i in indices_array:
            # Convert index to word
            mapped_text.append(self.index_to_word[i])
        
        return " ".join(mapped_text)
