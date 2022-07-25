"""
Shared Methods by two or more scripts.
"""

import gensim
import numpy as np


def print_done():
    """
    It is what it is
    """
    print('Done!\n')


def get_words_in_film(film, model):
    """
    Gets words from the film script file if they are in the model vocabulary
    :param film: name of film
    :param model: Word2vec model
    :return: list of words in a film
    """
    print("Getting words in film script")
    stop = len(film.columns)
    words = [film.columns[i] for i in range(2, stop) if (film.iat[0, i] == 1) and film.columns[i] in model.vocab]
    # words_in_film_vec = turn_words_to_vec(model, words)
    print_done()
    return words


def turn_words_to_vec(model, words):
    words_in_film_vec = []
    for word in words:
        words_in_film_vec.append(model.get_vector(word))
    words_in_film_vec = np.array(words_in_film_vec)
    return words_in_film_vec


def load_model():
    """
    Loads Google's word2vec model
    :return: loaded model
    """
    print('loading word2vec model, please wait...')
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print_done()
    return model
