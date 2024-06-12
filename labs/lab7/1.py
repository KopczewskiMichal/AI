import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def main():
  stop_words = stopwords.words('english')
  
  document: list = readDocument("article.txt")
  tokens = nltk.word_tokenize(document)
  print(f"Ilość słów po stokenizowaniu: {len(tokens)}")

  filtered_sentence = [w for w in tokens if not w.lower() in stop_words]
  print(f"Ilość słów po usunięciu stop_words: {len(filtered_sentence)}")
  
  stop_words.extend([',', '.', '“', '”', '’', "'s"])
  
  filtered_sentence = [w for w in tokens if not w.lower() in stop_words]
  print(f"Ilość słów po usunięciu dodanych stop_words: {len(filtered_sentence)}")
  
  lemmas = lemmatize(filtered_sentence)
  print(f"Ilość słów po lematyzacji: {len(lemmas)}")
  
  # count_words(lemmas)
  word_cloud(lemmas)
  
    
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
  
def word_cloud(tokens):
  filtered_text = " ".join(tokens)
    
  wordcloud = WordCloud(width = 800, height = 800, 
                  background_color ='white', 
                  min_font_size = 10).generate(filtered_text)

  plt.figure(figsize = (8, 8), facecolor = None) 
  plt.imshow(wordcloud) 
  plt.axis("off") 
  plt.tight_layout(pad = 0) 

  plt.show()




if __name__ == "__main__":
  main()