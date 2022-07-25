"""

Shared constants between two or more scripts of the project

"""

YEAR_RANGE_BETWEEN_FILMS = 5

FILM_NAME_AND_YEAR_RE_PATTERN = "([^,]+|\".*\"?),(19[0-9]{2}|20[0-9]{2}),"
TITLES_OF_ERA = ['Five years before film', 'Films in the same year', 'Five years after Film']

ENCODINGS = ['ascii', "utf-8", "utf-16", "utf-32"]

METADATA_TXT = 'movie_titles_metadata.txt'
SCRIPT_URLS_TXT = "raw_script_urls.txt"
WRITE_MODE = 'x'
FILM_INPUTS = ['the wizard of oz', "it's a wonderful life", '2001: a space odyssey',
               'jaws', 'rocky', 'top gun', 'hotel rwanda', 'taxi driver', 'the matrix', 'aliens', 'arcade',
               'the patriot', 'the jazz singer']

YEAR_OF_THE_FILM = 'Year of the film'
NAME_OF_THE_FILM = 'Name of the Film'
SCRIPT = 'script'
NAME = 'name'
YEAR = 'year'
BINARY_CSV = 'Binary.csv'
