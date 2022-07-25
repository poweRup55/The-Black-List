"""
THIS SCRIPT GRABS FILM SCRIPTS FROM THE WEB AND TOKENIZES AND LEMETIZE IT.

Watch presentation for more details
"""

import functools
import re
import string
import urllib
from os import path
from threading import Thread

import pandas as pd
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

import film_and_words
from shared_constants import ENCODINGS, YEAR_OF_THE_FILM, NAME_OF_THE_FILM, SCRIPT, NAME, YEAR, METADATA_TXT, \
    SCRIPT_URLS_TXT, WRITE_MODE

MIN_CHAR_SIZE_OF_SCRIPT = 25000  # Used to fish out scripts that are too short.


def regex_films():
    """
    Get the name, script url and year of a film, using regex, from metadata text files.
    """
    films = []
    with open(SCRIPT_URLS_TXT) as text:
        lines = text.readlines()
        for line in lines:
            x = re.match("m[0-9]* \\+\\+\\+\\$\\+\\+\\+ (.*) \\+\\+\\+\\$\\+\\+\\+ (.*)", line)
            films.append([x.group(1), x.group(2)])
    with open(METADATA_TXT) as text:
        lines = text.readlines()
        i = 0
        for line in lines:
            x = re.match("m[0-9]* \\+\\+\\+\\$\\+\\+\\+ (.*?) \\+\\+\\+\\$\\+\\+\\+ ([0-9]*)", line)
            films[i].append(x.group(2))
            i += 1
    return pd.DataFrame(films, columns=[NAME, SCRIPT, YEAR])


def script_grabber(film):
    """
    Get the script from a file or from the web.
    :param film: A Series obj of the film with - name, script, year
    :return: String of the script text
    """
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print('Reading ' + str(film[NAME]))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    script_file_name = str(film[NAME]).translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
    script_file = "scripts\\{name}.txt".format(name=script_file_name)
    # Check if file exist. If it does, return it as a string
    if path.exists(script_file):
        print('File exists! reading text file')
        f = open(script_file, 'rt')
        text = f.read().replace('\n', ' ').replace('-', ' ')
        f.close()
        return text

    print('File doesnt exist! getting script from web')
    # Get script from web
    script = str(film[SCRIPT])
    try:
        read_script_from_web = timeout(timeout=25)(read_from_web)
        text = read_script_from_web(script)
    except:
        print('Failed getting file from web!')
        return None

    print('Success pull')
    # write down text file.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    write_script_to_txt(script_file, text)
    return text


def write_script_to_txt(script_file, text):
    """
    Writes script to txt file
    :param script_file: Path to script file
    :param text: The script to write down
    """
    for encoding in ENCODINGS:
        # write down text file.
        f = open(script_file, WRITE_MODE, encoding=encoding)
        try:
            f.write(text)
        except UnicodeEncodeError:
            print('Error writing file!')
            f.close()
        else:
            f.close()
            break


def read_from_web(script):
    """
    Get the script from the script url
    :param script: url of the script
    :return: text string
    """
    print('Connecting...')
    response = urllib.request.urlopen(script)
    html = film_and_words.get_film_from_database()
    soup = BeautifulSoup(html, 'html5lib')
    script_txt = soup.get_text(strip=True)
    return script_txt


def tokenize_script(films):
    """
    Gets a script from the web and tokenizes the words in it.
    :param films: A Dataframe of films
    """
    stop_words = stop_words_no_punc()
    word_lematizer = WordNetLemmatizer()
    films_n_words = []
    for i in range(len(films)):
        film = films.iloc[i]
        script_txt = script_grabber(film)
        if script_txt != '' and script_txt != None and len(script_txt) > MIN_CHAR_SIZE_OF_SCRIPT:
            script_tokenizer(film, films_n_words, stop_words, script_txt, word_lematizer)
        else:
            print("Bad script or no script found!")
        print("\n")
    return films_n_words


def finalize_file(table):
    """
    Finalizes the binary
    :param table: A list of all words and films
    """
    print("Finished pooling scripts!", "Concating Dataframes")
    table = pd.concat(table, ignore_index=True)
    print("Finished Concating Dataframes", "\n")
    print("Filling NaN with FALSE")
    table.fillna(0, inplace=True)
    print("Finished filling NaN with FALSE!")
    print("Saving file!")
    save_csv(table)
    print("Finito!!!")


def save_csv(table):
    """
    Writes table to csv
    :param table: A Dataframe of all words and films
    """
    try:
        table.to_csv("Binary.csv", index=False)
    except PermissionError:
        print("Please close file!")
        save_csv(table)


def script_tokenizer(film, films_n_words, stop_words, script_txt, word_lematizer):
    """
    Tokenizes a script
    :param film: A dataframe of the film
    :param films_n_words: A list of films and their tokenized word
    :param stop_words: A set of stopwords
    :param script_txt: The text of the script
    :param word_lematizer: Lematizer obj
    """
    word_tokens = word_tokenize(script_txt)
    word_tokens = more_removal(word_tokens)
    word_tokens = list({word_lematizer.lemmatize(w) for w in word_tokens if w not in stop_words})
    df = pd.DataFrame(1, columns=[NAME_OF_THE_FILM, YEAR_OF_THE_FILM] + word_tokens, index=[0])
    df[NAME_OF_THE_FILM] = film['name']
    df[YEAR_OF_THE_FILM] = film['year']
    films_n_words.append(df)
    print("Script added to database!")


def stop_words_no_punc():
    """
    :return: Stop words with no punctuation and with punctuation
    """
    stop_words = set(stopwords.words('english'))
    new_stopwords = set()
    for stop_word in stop_words:
        new_stopwords.add(stop_word)
        new_stopwords.add(stop_word.translate(str.maketrans('', '', string.punctuation)))
    return new_stopwords


def more_removal(word_tokens):
    """
    Removes digits and single letter words
    :param word_tokens: tokens of words
    :return: words after removal
    """
    good_words = set()
    for word in word_tokens:
        if not re.search('\d+', word):
            if len(word) > 1:
                good_words.add(word)
    return good_words


def timeout(timeout):
    """
    Timesout if a function takes too long
    :param timeout: Time to timeout
    """

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


def write_binary_file():
    """
    Main
    """
    # nltk.download()
    films = regex_films()
    films_n_words = tokenize_script(films)
    finalize_file(films_n_words)


if __name__ == '__main__':
    write_binary_file()
