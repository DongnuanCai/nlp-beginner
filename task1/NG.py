import numpy as np
import re
from string import punctuation as punct
from scipy.sparse import lil_matrix

class Ngram:
    def __init__(self,n):
        self.n=n
        self.feature_map={}
    
    def N_Gram(self,data_x):
        for sen in data_x:
            sen = ["<s>"]+re.split("[{} ]".format(punct),sen)+["</s>"]
            for i in range(len(sen)-self.n+1):
                feature = "_".join(sen[i:i + self.n])
                if feature not in self.feature_map:
                        self.feature_map[feature] = len(self.feature_map)
            
        n=len(data_x)
        m=len(self.feature_map)
        ngram_feature=ngram_feature = lil_matrix((n, m), dtype=np.int8)
        for idx,sen in enumerate(data_x):
            sen = ["<s>"]+re.split("[{} ]".format(punct),sen)+["</s>"]
            for i in range(len(sen)-self.n+1):
                feature = "_".join(sen[i:i + self.n])
                if feature in self.feature_map:
                        ngram_feature[idx, self.feature_map[feature]] += 1
                        
        return ngram_feature
                
                
            