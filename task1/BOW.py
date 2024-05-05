import numpy as np
import re
from string import punctuation as punct
from scipy.sparse import lil_matrix

class BagofWords:
    def __init__(self):
        self.vocab={}
        
    def transform(self,data_x):
        for sen in data_x:
            words = re.split("[{} ]".format(punct),sen)
            for w in words:
                if w not in self.vocab:
                    self.vocab[w]=len(self.vocab)
                    
        vocab_size=len(self.vocab)
        bag_of_word=lil_matrix((len(data_x),vocab_size), dtype=np.int8)
        for i,sen in enumerate(data_x):
            words = re.split("[{} ]".format(punct),sen)
            for w in words:
                if w in self.vocab:
                    bag_of_word[i,self.vocab[w]]+=1
        return bag_of_word
                
                
                
            
        
        
        
    
        
    