
from nltk.stem.snowball import SnowballStemmer
import string

stemmer = SnowballStemmer("english")
print(stemmer.stem("stephanie"))