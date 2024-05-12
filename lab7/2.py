import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te

def hotels():
  sent_analz = SentimentIntensityAnalyzer()
  
  positive = readDocument("./positive_opinion.txt")
  print(f"Wynik dla pozytywnej opinii {sent_analz.polarity_scores(positive)}")
  print(f"Emocje dla pozytywnej opinii {te.get_emotion(positive)}")
  
  negative = readDocument("./negative_opinion.txt")
  print(f"Wynik dla negatywnej opinii {sent_analz.polarity_scores(negative)}")
  print(f"Emocje dla negatywnej opinii {te.get_emotion(negative)}")
  
  
  
  
def readDocument(path: str) -> list:
  document = open(path, "r").read()
  return document

if __name__ == "__main__":
  hotels()
  
# TripAdvisor w podejrzany sposób zarządza opiniami, hotele mające ponad 6 tysięcy opinii często mają ok. 4 negatywnych. Odpowiednio mocno złą opinię uzyskałem za pomocą chat GPT 3.