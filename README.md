# The Black List


## There are two main scripts

### film_and_words.py

This is the first attempt to find a unique film using its script.
This program takes an input film script and performs several tasks on it.
Firstly, it retrieves all the words that appeared in the movie's script.
Then, it selects the 500 most frequently used words in all the movies that were made five years prior to the input film, as well as those made five years after, and those that were released in the same year.
Finally, it uses Google's pre-trained word2vec model to convert all the words into vectors and uses PCA (Principal Component Analysis) to show a grath.
This program provides insights into how the inputted movie changed other movies scripts. The resultes were inconclusive.

### words_in_film

This is the second attempt to find a unique film using its script.
This program helps find the first movie in which a specific word was used, or a similar word.
It then uses Google's pre-trained word2vec model to find a set of words that are similar to the input word.
From this set, the program extracts the 20 most frequently used words in two timeframes: 5 years before the movie's release and 5 years after the movie's release.
This can be useful in exploring the usage of specific words in movies and how one movie could have helped change the direction of the film industry.

## For more inforamtion and to watch some example grath, you can watch this slideshow:
https://docs.google.com/presentation/d/13ZqnHZmzhomUFCEgutf4yKX-PP2-eIQ5/edit#slide=id.p20
