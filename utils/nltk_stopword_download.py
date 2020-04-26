import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')


stopwords = stopwords.words("english")

stemmer = PorterStemmer()
