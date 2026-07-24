import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# buscar sentidos de una palabra
synsets = wn.synsets('poppy', pos=wn.NOUN)

for s in synsets:
    print(s.name())
    print("def:", s.definition())
    #print("ej:", s.examples())
    #print("lemmas:", [l.name() for l in s.lemmas()])
    print("------")