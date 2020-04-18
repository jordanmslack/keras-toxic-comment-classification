import re 
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model


lemm = WordNetLemmatizer()


def clean_text(sentence, lemm=lemm):
    
    """
    This function processes the senteence passed to it by replacing all symbols, removing line breaks, 
    converting the text to lower case, tokenizing the sentence and removing english stop words.
    
    :param sentence:
        A raw, plain text english sentence
        
    :return:
        A cleaned, processed sentence
    """
    
    sentence = re.sub(r'[^\w]', ' ', sentence.lower())
    sentence = word_tokenize(sentence)
    sentence = ' '.join(lemm.lemmatize(i, pos="v") for i in sentence if not i in stopwords.words('english'))
    
    return sentence


def parse_glove_file(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def plot_polarity(df, lowercase=True, max_features=50000, ngram_range=(1,1)):
    
    """
    A function that processes data from a pandas dataframe, parses the text and creates a diagram displaying
    the distribution of words/ngrams against their concentration in the overall data set.
    
    :param df:
        A pandas dataframe
    :param lowercase:
        A boolean value indicating whether text should be processed as all lowercase or as is
    :param max_features:
        The maximum number of featurees that should be considered for vectorization 
    :param ngram_range:
        The lower and upper boundary of the range of n-values for different n-grams to be extracted
        
    :return:
        a matplot lib diagram of top and bottom samples of the TFIDF vectorized ngrams
    """
    
    tfidf = TfidfVectorizer(lowercase=lowercase, max_features=max_features, ngram_range=ngram_range, )
    lr = LogisticRegression(solver='liblinear')
    p = make_pipeline(tfidf, lr)
    p.fit(df['comment_text'].values, df['toxic'].values)

    rev = sorted({v: k for k,v in p.steps[0][1].vocabulary_.items()}.items())
    polarity = pd.DataFrame({'coef': p.steps[1][1].coef_[0]}, 
                            index = [i[1] for i in rev]).sort_values('coef')

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1,2,1)
    polarity.tail(25).plot(kind='barh', color='orange', ax=ax)
    ax = plt.subplot(1,2,2)
    polarity.head(25).plot(kind='barh', ax=ax)
    
    
def plot_class_distribution(df, title, xlabel, ylabel):
    
    """
    Takes data from a pandas data frame, parses class distribution and plots in a matplot/seaborn diagram.
    
    :param df:
        A pandas dataframe
    :param title:
        title to be assigned to the plot
    :param xlabel:
        label to be assigned to the x axis of the plot
    :param ylabel:
        label to be assigned to the y axis of the plot
        
    :return:
        a diagram of class distribution
    
    """
    
    distribution = df.iloc[:, 2:].sum().sort_values()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(distribution.index, distribution.values, alpha=0.8)
    
    plt.title(title)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)

    rects = ax.patches
    labels = distribution.values

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

    plt.show()
    

def model_outputs(num_words, embed_size, embedding_matrix, inputs):
    
    """
    Creates the strucutures and outputs used in the creation of the Keras model.
    
    :param num_words:
        The maximum number of words to be used in the model 
    :param embed_size:
        The size of embedding array being passed
    :param embedding_matrix:
        A matrix providing the weights 
    :param inputs:
        Input labels
        
    :return output:
        model outputs prepped and configured
        
    """
    
    outputs = Embedding(num_words, embed_size, weights=[embedding_matrix])(inputs)
    outputs = Bidirectional(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(outputs)
    outputs = GlobalMaxPool1D()(outputs)
    outputs = Dense(50, activation="relu")(outputs)
    outputs = Dropout(0.1)(outputs)
    outputs = Dense(6, activation="sigmoid")(outputs)

    return outputs
