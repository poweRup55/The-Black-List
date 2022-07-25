"""
Second (and a more successful) attempt to find a unique film using its script

Watch presentation for more details
"""

import string
import sys
from os import path

import matplotlib.pyplot as plt
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.decomposition import PCA

from color import Color
from shared_constants import BINARY_CSV, YEAR_RANGE_BETWEEN_FILMS, YEAR_OF_THE_FILM, NAME_OF_THE_FILM
from shared_methods import turn_words_to_vec, load_model

WORDS_COUNT_IN_GRAPH_MAX = 20  # Maximum number of words before or after an inputted film.


def before_and_after_film_words(after_film, before_films, model, similar_words):
    """
    Words from scripts before and after a given script
    :param after_film: words from scripts before a given script
    :param before_films: words from scripts after a given script
    :param model: word2vec model
    :param similar_words: similar words to the inputted words found
    :return: all the words before and after the film
    """
    words_count = min(WORDS_COUNT_IN_GRAPH_MAX, len(similar_words))
    word_before_films = [str(x) for x in
                         before_films.sum().iloc[3:].sort_values(ascending=False).index.tolist() if
                         x in model.vocab][:words_count]
    word_after_films = [str(x) for x in
                        after_film.sum().iloc[3:].sort_values(ascending=False).index.tolist() if
                        x in model.vocab][:words_count]
    return word_after_films, word_before_films


def initialize_model():
    """
    Initializes model and dependencies
    :return: words found in the scripts, word2vec model, word lemerizer.
    """
    model = load_model()
    word_lematizer = WordNetLemmatizer()
    with open(BINARY_CSV, encoding='utf-8') as binary:
        words_in_scripts = binary.readline().split(',')
    return words_in_scripts, model, word_lematizer


def graph_by_word(first_film, first_film_year, model, word_after_films, word_before_films, word_input):
    """
    PCA and graph of the findings.
    """
    print('Graphing it')
    transformer = PCA(n_components=2)
    name = "results\\words\\" + str(YEAR_RANGE_BETWEEN_FILMS) + '_' + word_input.translate(
        str.maketrans('', '', string.punctuation)).replace(' ', '_')
    if path.exists(name + '.txt'):
        with open(name + '.txt', 'wt') as f:
            f.write("Before : " + str(word_before_films))
            f.write('\nAfter : ' + str(word_after_films))
    else:
        with open(name + '.txt', 'xt') as f:
            f.write("Before : " + str(word_before_films))
            f.write('\nAfter : ' + str(word_after_films))

    vectors = turn_words_to_vec(model, word_before_films + word_after_films + [word_input])
    words_vec_new = transformer.fit_transform(vectors)
    word_before_films = words_vec_new[:WORDS_COUNT_IN_GRAPH_MAX]
    word_after_films = words_vec_new[WORDS_COUNT_IN_GRAPH_MAX:-1]
    word_input_vec = words_vec_new[-1]
    show_graph_by_word(first_film, first_film_year, name, word_after_films, word_before_films, word_input,
                       word_input_vec)


def show_graph_by_word(first_film, first_film_year, name, word_after_films, word_before_films, word_input,
                       word_input_vec):
    """
    Graph all the data and export.
    """
    fig = plt.figure()
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    # ax1.axis('off'), ax2.axis('off')
    fig.suptitle(
        '"' + string.capwords(word_input) + '"' + " was first used by the film " + string.capwords(
            first_film) + ' in ' + str(
            first_film_year), fontsize=12)
    ax1.scatter(word_before_films[:, 0], word_before_films[:, 1])
    ax1.scatter(word_input_vec[0], word_input_vec[1], c='r')
    ax1.set_title(str(YEAR_RANGE_BETWEEN_FILMS) + " years before movie")
    ax2.scatter(word_after_films[:, 0], word_after_films[:, 1])
    ax2.scatter(word_input_vec[0], word_input_vec[1], c='r')
    ax2.set_title(str(YEAR_RANGE_BETWEEN_FILMS) + " years movie movie")
    plt.savefig(
        name + '.png', dpi=None, facecolor='w',
        edgecolor='b', transparent=True)
    plt.show()
    plt.clf()
    plt.close(fig)


def get_similar_words_from_model(words_in_scripts, model, word_input, word_lematizer):
    """
    Gets similar words to the word input
    :param words_in_scripts: binary of words in all the database scripts
    :param model: word2vec model
    :param word_input: input word from user
    :param word_lematizer: word lematizer
    :return: column titles from database, list of similar words and the inputted word (might have changed)
    """
    print('Finding similar words to {} in google database'.format(word_input))
    similar_words = model.similar_by_word(word_input, 1000)
    similar_words = {word_lematizer.lemmatize(similar_word[0].translate(str.maketrans('', '', string.punctuation))) for
                     similar_word in similar_words}
    similar_words = [similar_word for similar_word in similar_words if similar_word in words_in_scripts]
    print('Found {} similar words'.format(len(similar_words)))
    column_titles = [NAME_OF_THE_FILM, YEAR_OF_THE_FILM]
    if word_input not in words_in_scripts:
        word_input = similar_words[0]
        similar_words = similar_words[1:]
        print('input word was not found om film database, but similar word was found!')
        print('The new word is {}'.format(word_input))
    column_titles.append(word_input)
    return column_titles, similar_words, word_input


def get_film_with_word(first, table, sorted_table, word_input):
    """
    Something that I did years ago. not sure what I did but it works.
    """
    if first >= len(sorted_table):
        raise ValueError
    first_film = sorted_table.iat[first, 0]
    first_film_year = sorted_table.iat[first, 1]
    five_before_films = table[table[YEAR_OF_THE_FILM] < first_film_year]
    five_before_films = five_before_films[
        five_before_films[YEAR_OF_THE_FILM] > first_film_year - YEAR_RANGE_BETWEEN_FILMS]
    five_after_film = table[table[YEAR_OF_THE_FILM] < first_film_year + YEAR_RANGE_BETWEEN_FILMS]
    five_after_film = five_after_film[five_after_film[YEAR_OF_THE_FILM] > first_film_year]
    if len(five_before_films) <= 2:
        first += 1
        return get_film_with_word(first, table, sorted_table, word_input)
    if len(five_before_films) <= 2:
        first += 1
        return get_film_with_word(first, table, sorted_table, word_input)
    print(
        'The word {bold}{inputW}{end} was in the film {bold}{film}{end} that came in {bold}{year}{end}'.format(
            inputW=word_input, film=string.capwords(first_film), year=first_film_year, bold=Color.RED,
            end=Color.END))
    return first_film, five_after_film, five_before_films, first_film_year


def main_loop():
    """
    Main loop of program - given a word - finds the first film that used said word and
    shows a graph of words used in films 5 years before and after
    """
    words_in_scripts, model, word_lematizer = initialize_model()
    while True:
        word_input = input("Enter a word or write 'q' to exit: ").lower()
        if word_input == 'q':
            break
        print('\n')
        if word_input in model.vocab:  # if inputted word is in word2vec model, get similar words to it.
            films_with_similar_words, similar_words, word_input = get_similar_words_from_model(words_in_scripts, model,
                                                                                               word_input,
                                                                                               word_lematizer)
            # Get only films with the similar words from the database and sort it.
            similar_words_of_movie = pd.read_csv(BINARY_CSV, usecols=films_with_similar_words + similar_words)
            sorted_table = similar_words_of_movie.sort_values(YEAR_OF_THE_FILM, ignore_index=True)
            sorted_table = sorted_table[sorted_table[word_input] == 1]
            # Finds the movie that the word was first appeared and the films that came before and after it.
            try:
                first_film, films_after, films_before, first_film_year = get_film_with_word(0, similar_words_of_movie,
                                                                                            sorted_table, word_input)
            except ValueError as e:
                print("Didn't find a suitable movie for this word")
                continue
            word_after_films, word_before_films = before_and_after_film_words(films_after, films_before, model,
                                                                              similar_words)

            graph_by_word(first_film, first_film_year, model, word_after_films, word_before_films, word_input)
        else:
            print(word_input + ' was not found in google database\n')


if __name__ == '__main__':
    main_loop()
    sys.exit(0)
