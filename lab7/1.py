import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def main():
  stop_words = stopwords.words('english')
  stop_list = [',', '.', '“', '”', '’', "'s"]
  stop_words.extend(stop_list)
  
  document: list = readDocument("article.txt")
  tokens = nltk.word_tokenize(document)
  print(f"Ilość słów po stokenizowaniu: {len(tokens)}")

  filtered_sentence = [w for w in tokens if not w.lower() in stop_words]
  print(f"Ilość słów po usunięciu stop_words: {len(filtered_sentence)}")
  lemmas = lemmatize(filtered_sentence)
  print(f"Ilość słów po lematyzacji: {len(lemmas)}")
  
  count_words(lemmas)
  
    
def lemmatize(filtered_sentence: list) -> list:
  wnl = WordNetLemmatizer()
  lemmas = [wnl.lemmatize(word, pos="a") for word in filtered_sentence]
  # for words in filtered_sentence:
    # print(words + " ---> " + wnl.lemmatize(words, pos="a"))
  return lemmas
  

def readDocument(path: str) -> list:
  document = open(path, "r").read()
  return document

def count_words(tokens):
  fdist1 = nltk.FreqDist(tokens)
  print (fdist1.most_common(10))
  fdist1.plot(20)




if __name__ == "__main__":
  main()