import pandas as pd
import string
from textacy import preprocess
from gensim.models import Phrases
from gensim.models.word2vec import Word2Vec
import re
import sys

# A custom function to clean the text before sending it into the vectorizer
def cleanTextSpacy(case):

    case = case.encode('ascii', 'ignore').decode('ascii') # case.encode('utf-8')

    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = str(case).translate(replace_punctuation)

    text = preprocess.preprocess_text(text, lowercase=True, no_urls=True, no_emails=True, no_phone_numbers=True,
                                      no_numbers=True, no_currency_symbols=True, no_contractions=True,
                                      no_accents=True)
    text = preprocess.normalize_whitespace(text)
    text = re.sub("weather related yes", '. This claim is related to weather.', text)
    text = re.sub("weather related y", '. This claim is related to weather.', text)
    text = re.sub(r"weather related no", '', text)
    text = re.sub(r"weather related n", '', text)
    text = re.sub(r"weather related", '', text)
    text = re.sub(r"additional involved parties none", '', text)
    text = re.sub(r"additional involved parties", '.', text)
    # text = str(text)
    # print text.split()
    return text.split()


data = pd.read_csv('../release10/descrip_208644.csv', encoding='latin_1') #, header=None, , delimiter='\t'
print data.columns
data.drop_duplicates(subset='Original Description', inplace=True) #
print len(data)

data['Original Description'] = data['Original Description'].fillna('')
# print str(len(data)), 'rows loaded after removing duplicates.'
cleanedNotes = data['Original Description'].apply(cleanTextSpacy)
# equian_ids = data['Equian_ID'].values

# bigram_transformer = Phrases(cleanedNotes)
# trigram_transformer = Phrases(bigram_transformer[cleanedNotes])

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 2   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# model = Word2Vec(trigram_transformer[bigram_transformer[cleanedNotes]], workers=num_workers, \
#             size=num_features, min_count = min_word_count, \
#             window = context, sample = downsampling, seed = 786)
# model = Word2Vec(bigram_transformer[cleanedNotes], workers=num_workers, \
#            size=num_features, min_count = min_word_count, \
#            window = context, sample = downsampling, seed = 786)
model = Word2Vec(cleanedNotes, workers=num_workers, \
           size=num_features, min_count = min_word_count, \
           window = context, sample = downsampling, seed = 786)

print model.syn0.shape

# to trim unneeded model memory = use (much) less RAM
model.init_sims(replace=True)
model_name = 'word2vec_ccc'
model.save(model_name)