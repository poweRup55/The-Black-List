"""
First (and worst) attempt to find a unique film using its script.

Watch presentation for more details
"""

import re
import string
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from shared_constants import YEAR_RANGE_BETWEEN_FILMS, FILM_NAME_AND_YEAR_RE_PATTERN, TITLES_OF_ERA, \
    YEAR_OF_THE_FILM, NAME_OF_THE_FILM, BINARY_CSV
from shared_methods import print_done, get_words_in_film, turn_words_to_vec, load_model

BINARY_ENCODING = 'utf-8'
NUM_OF_MOST_USED_WORDS = 500


def get_same_era_films(film_name, film_year):
    """
    Given a film and the year it was produced, returns from the database only films from
    year_input - YEAR_RANGE_BETWEEN_FILMS <= year_input <= year_input + YEAR_RANGE_BETWEEN_FILMS
    and sets the dataframe of it - with the words that are in the scripts.
    :param film_name: name of the film
    :param film_year: year of the film
    :return: a binary of the films within the range of years and a binary of the film itself.
    """
    print('Getting films from same era...')
    columns, films_to_check = get_same_era_films_from_binary(film_year)
    print_done()
    print('Setting Dataframe...')
    films = pd.DataFrame(films_to_check, columns=columns)
    five_before_films = films[films[YEAR_OF_THE_FILM] < film_year]
    five_after_film = films[films[YEAR_OF_THE_FILM] > film_year]
    same_year_films = films[films[YEAR_OF_THE_FILM] == film_year]
    same_year_films = same_year_films[same_year_films[NAME_OF_THE_FILM] != film_name]
    film = films[films[NAME_OF_THE_FILM] == film_name]
    print_done()
    return five_before_films.reset_index(), same_year_films.reset_index(), five_after_film.reset_index(), film.reset_index()


def get_same_era_films_from_binary(year_input):
    """
    Finds a range of film from the database that were produced in
    year_input - YEAR_RANGE_BETWEEN_FILMS <= year_input <= year_input + YEAR_RANGE_BETWEEN_FILMS
    returns their name, year, and the number of times a word has beem written in its script.
    :param year_input: >= 1900
    :return: column line in the binary file, a list of films of the same era with their tokenized and lemetized words.
    """
    with open(BINARY_CSV, encoding=BINARY_ENCODING) as binary:
        columns_in_binary_file = binary.readline().split(',')
        min_year = year_input - YEAR_RANGE_BETWEEN_FILMS
        max_year = year_input + YEAR_RANGE_BETWEEN_FILMS
        film_line_in_file = binary.readline()
        films_same_era = []
        while film_line_in_file is not None and film_line_in_file != "":
            regex_of_film_file_line = re.match(FILM_NAME_AND_YEAR_RE_PATTERN, film_line_in_file)
            film_name_same_era = regex_of_film_file_line.group(1)
            film_year_same_era = int(regex_of_film_file_line.group(2))
            if min_year <= film_year_same_era <= max_year:
                regex_of_film_file_line = re.match(FILM_NAME_AND_YEAR_RE_PATTERN + "(.*)", film_line_in_file)
                words_in_script = [float(num) for num in regex_of_film_file_line.group(3).split(',')]
                films_same_era.append(
                    [film_name_same_era, film_year_same_era] + words_in_script)
            film_line_in_file = binary.readline()
    return columns_in_binary_file, films_same_era


def get_film_from_database(film_name_input):
    """
    Gets the film and film year from the database binary.
    Each line in database contains a film name and its production year.
    :param film_name_input: name of the film
    :return: if found, return name of the film and the production year. Else return None,None
    """
    with open(BINARY_CSV, encoding=BINARY_ENCODING) as binary:
        binary.readline()
        film_line_in_file = binary.readline()
        cand = []
        while film_line_in_file is not None and film_line_in_file != '':
            x = re.match(FILM_NAME_AND_YEAR_RE_PATTERN, film_line_in_file)
            film_name = x.group(1)
            if film_name.lower() == film_name_input.lower():
                print('Found!\n')
                film_year = int(x.group(2))
                return film_name, film_year
            elif film_name_input.lower() in film_name.lower():
                cand.append(film_name)
            film_line_in_file = binary.readline()
        return film_not_found(cand)


def film_not_found(cand):
    """
    If a film was not found in films database.
    :param cand: candidates of film that the user might have ment.
    :return: None, None
    """
    print("Film was not found!")
    if len(cand) != 0:
        print("Did you mean:\n")
        print(*cand, sep='\n')
        print('\n')
    return None, None


def main_loop():
    """
    Main loop of software
    """
    model = load_model()
    while True:
        film_name_input = input("Enter a film name or write 'q' to exit: ")
        if film_name_input == 'q':
            break
        film_name, film_year = get_film_from_database(film_name_input)
        if film_name is None:
            continue
        films_by_era = get_same_era_films(film_name, film_year)
        film = films_by_era[-1]
        words_in_film = get_words_in_film(film, model)
        most_used_words_by_era = get_most_used_words_by_era(films_by_era, model)
        PCA_projection_n_graph(model, most_used_words_by_era, words_in_film, film_name)


def get_most_used_words_by_era(films_by_era, model):
    """
    returns a list of most used words by era.
    :param films_by_era: a dataframe of films by era
    :param model: word2vec model
    :return: a list of most used words by era.
    """
    most_used_words_by_era = []
    for i in range(len(films_by_era) - 1):
        films_of_an_era = films_by_era[i]
        print('Getting most used words in: ', TITLES_OF_ERA[i])
        words = [str(x) for x in films_of_an_era.sum().iloc[3:].sort_values(ascending=False).index.tolist() if
                 x in model.vocab]
        most_used_words_by_era.append(words[:NUM_OF_MOST_USED_WORDS])
    return most_used_words_by_era


def PCA_projection_n_graph(model, most_used_words_by_era, words_in_film_vec, film_name):
    """
    PCA projcets the dataframe and then saves the graphs
    :param model: word2vec model
    :param most_used_words_by_era:  a list of most used words by era.
    :param words_in_film_vec: Words that are in the given films - as a vector
    :param film_name: name of the inputted film
    """
    transformer = PCA(n_components=2)
    vectors = turn_words_to_vec(model, words_in_film_vec)
    sizes = [words_in_film_vec.shape[0]]
    for freq_words in most_used_words_by_era:
        words_in_era_vec = turn_words_to_vec(model, freq_words)
        sizes.append(words_in_era_vec.shape[0])
        vectors = np.concatenate([vectors, words_in_era_vec], axis=0)
    words_vec_new = transformer.fit_transform(vectors)
    words_in_film_vec_new = words_vec_new[:sizes[0]]
    so_far = sizes[0]
    for i in range(len(most_used_words_by_era)):
        most_used_words_by_era[i] = words_vec_new[so_far:so_far + sizes[i + 1]]
        so_far += sizes[i + 1]
    show_graph_of_films(film_name, most_used_words_by_era, words_in_film_vec_new)


def show_graph_of_films(film_name, most_used_words_by_era, words_in_film_vec_new):
    """
    Does the graphing of the words in the scripts.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.axis('off')
    axs = [fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
    fig.suptitle(string.capwords(film_name) + " Word2Vec Projection", fontsize=16)
    ax1.scatter(words_in_film_vec_new[:, 0], words_in_film_vec_new[:, 1], c='#3d91e0', marker='^', s=0.1)
    ax1.set_title("Film projection")
    for i in range(len(most_used_words_by_era)):
        axs[i].scatter(words_in_film_vec_new[:, 0], words_in_film_vec_new[:, 1], c='#3d91e0', marker='^', s=0.1)
        axs[i].scatter(most_used_words_by_era[i][:, 0], most_used_words_by_era[i][:, 1], c='#f51720', marker='o', s=1)
        axs[i].axis('off')
        axs[i].set_title(TITLES_OF_ERA[i])
    plt.savefig("results\\" + film_name.translate(str.maketrans('', '', string.punctuation)).replace(' ', '_') + '.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main_loop()
    sys.exit(0)
