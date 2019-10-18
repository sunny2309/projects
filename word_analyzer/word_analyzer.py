import pandas as pd
from collections import Counter
import math

class Preprocessor(object):
    def __init__(self):
        self.book_content = None
        
    def __str__(self):
        return self.book_content
    
    def clean(self):
        ## Below list let us keep only alphabets and numbers and spacer
        chars_numbers_space_tab = ['a','b','c','d','e','f','g','h','i','j','k','l','m',\
                                    'n','o','p','q','r','s','t','u','v','w','x','y','z',\
                                    '0','1','2','3','4','5','6','7','8','9', ' ', '\t']
        if not self.book_content:
            return 1
        else:
            total_corpus = []
            for char in self.book_content:
                if char in ['-','_', "\n", '—']: # If character is new line or dash or underscore then we add space
                    total_corpus.append(' ')
                elif char.lower() in chars_numbers_space_tab: ## We take only alphabets, numbers and space
                     total_corpus.append(char.lower())
            print('Total Character : %d'%len(total_corpus))
            return ''.join(total_corpus) 
            
    def read(self, text_name):
        self.book_content = open(text_name, encoding='utf-8').read()        
                
class WordAnalyzer(object):
    def __init__(self):
        self.word_count = None
    
    def __str__(self):
        final_str = ''
        for key, val in self.word_count.items(): # @3 create big string with new line after each key: val
            final_str = final_str + '%s : %d\n'%(key, val)
        return final_str
            
    def analyse_words(self, book_text):
        # We only keep string which has some characters. Counter returns dictionary with count of each word as value and word as key.
        self.word_count = Counter([word.strip() for word in book_text.split(' ') if word.strip()]) 
    
    def get_word_frequency(self):
        freq_dictionary = {}
        total_words = sum(list(zip(*self.word_count.items()))[1]) ## We get count of total words in document
        for key, val in self.word_count.items(): 
            freq_dictionary[key] = val / total_words # Calculating freq
        return freq_dictionary
        
                                
class IDF(object):
    def __init__(self):
        self.data = pd.DataFrame([])
        
    def load_frequency(self, book_frequency, book_title):
        if isinstance(self.data, pd.DataFrame):
            temp = self.data.T # We take transpose of original dataframe so words becomes index
            temp2 = pd.DataFrame([list(book_frequency.values())], columns = book_frequency.keys()) # We create new data frame for new book
            temp2['index'] = [book_title]
            temp2 = temp2.set_index('index')
            print(book_title + ' : ' + str(temp2.shape))
            out = temp.join(temp2.T, how='outer') ## We join both dataframe with outer join so that all words from both are kept with values
            self.data = out.T ## We then do transpose again to put word as column again.
            print('Total DF Shape : ',self.data.shape)
        else:
            data = pd.DataFrame([list(book_frequency.values())], columns = book_frequency.keys())
            data['index'] = [book_title]
            self.data = data.set_index('index')
            print(book_title + ' : ',str(self.data.shape))
        
    def getIDF(self, term):
        D = self.data.shape[0]
        #print(D)
        N = self.data[[term]].dropna(how='any').shape[0]
        #print(N)
        idf = 1 + math.log(D / (1+N))
        return idf
        
                        
def choice(term, documents):
    highest_doc, highest_tf_idf = None, 0
    if term in documents.data.columns:
        all_docs_with_term = documents.data[[term]].dropna(how='any')
        #print(all_docs_with_term)
        idf_val = documents.getIDF(term)
        print('IDF Val : %s'%str(idf_val))
        for file_name, tf in zip(all_docs_with_term.index, all_docs_with_term[term]):
            #print(tf)
            tf_idf = tf*idf_val
            if tf_idf > highest_tf_idf:
                highest_tf_idf = tf_idf
                highest_doc = file_name
    return highest_doc, highest_tf_idf
    
            
if __name__ == '__main__':
    idf = IDF()
    for file_name in ['11-0.txt', '84-0.txt', '1342-0.txt','1661-0.txt','1952-0.txt', 'pg16328.txt']:
        processor = Preprocessor()
        processor.read(file_name)
        cleaned_content = processor.clean()
        with open(file_name.split('.')[0] + '_clean.txt','w') as f:
            f.write(cleaned_content)
        word_analyzer = WordAnalyzer()
        word_analyzer.analyse_words(cleaned_content)
        with open(file_name.split('.')[0] + '_counts.txt','w') as f:
            f.write(str(word_analyzer))
        freq_dict = word_analyzer.get_word_frequency()
        with open(file_name.split('.')[0] + '_freq.txt','w') as f:
            f.write(str(freq_dict))
        idf.load_frequency(freq_dict, file_name)
    
    idf.data.to_csv('IDF.csv')
    
    with open('idf.txt','w') as f:
        f.write(idf.data.to_string(justify='right' , line_width=80))
    
    print('abcs : ', choice('abcs', idf))
    print('announce : ', choice('announce', idf))
    
    print('project : ',idf.getIDF('project'))
