from collections import Counter;
import time
from gtts import gTTS
import os
import threading
import requests
import json

class History:
  MAX_LENGTH = 10
  TIME_BEETWEN_REACTIONS = 10
  def __init__(self):
    self.last_reaction = time.time()
    self.history_arr = [] # przechowujemy historię jako listę int
    self.emotion_dict = {0: 'angry', 1: 'contempt', 2:'disgust', 3:'fear', 4:'happy', 5:'neutral', 6:'sad', 7:'suprise'}


  def _insert_prediction(self, prediction: int):
    if len(self.history_arr) >= self.MAX_LENGTH:
      self.history_arr.pop(0)
    self.history_arr.append(prediction)
  

  def _predict_longer_emotion(self, reaction: callable):
    self.counter = Counter(self.history_arr)
    most_frequent, count = self.counter.most_common(1)[0]
    if count > self.MAX_LENGTH * 0.5:
      reaction(most_frequent)

  def _print_emotion(self, emotion_index):
    print(f"You are: {emotion_index}")


  def handle_emotion(self, emotion_index):
    self._insert_prediction(emotion_index)
    act_time = time.time()
    if len(self.history_arr) == self.MAX_LENGTH \
      and act_time - self.TIME_BEETWEN_REACTIONS > self.last_reaction:
      self.last_reaction = act_time
      # self._predict_longer_emotion(self._print_emotion)
      self._predict_longer_emotion(self.audio_reaction)

  
  def audio_reaction(self, emotion):
    match emotion: 
      case "angry":
        mytext = "Keep calm and code on with Python"
      case "contempt":
          mytext = "Oops, my bad! I'll just go hide in the corner."
      case "disgust":
          mytext =  "Ew, gross! Let's change the topic."
      case "fear":
          mytext = "Don't worry, I've got a blanket and some hot cocoa to keep you safe!"
      case "happy":
          mytext = "Yay, you're as happy as a unicorn in a candy store!"
      case "neutral":
          mytext = "Hey, life's like a cup of coffee - it's what you make of it!"
      case "sad":
          mytext = self.get_joke()
      case "suprise":
          mytext = "Well, that came out of left field! Let's roll with it!"

    t = threading.Thread(target=self._read_text, args=(mytext,))
    t.start()
    
  def _read_text(self, text: str):
    language = 'en'
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("./main.mp3")
    os.system("afplay main.mp3")

  def get_joke(self) -> str:
    try:
      response = requests.get('https://v2.jokeapi.dev/joke/Programming?blacklistFlags=religious,political,racist,sexist&type=single')
      if response.status_code == 200:
        joke = json.loads(response.content)['joke']
        return joke
      else:
        return f"Wystąpił problem podczas wysyłania zapytania. Kod odpowiedzi: {response.status_code}"
    except Exception as e:
      return "Wystąpił błąd podczas pobierania żartu"



if __name__ == '__main__':
  myHistory = History()
  start = time.time()
  print(myHistory.get_joke())

  print(f"Executed in {start - time.time()}s")