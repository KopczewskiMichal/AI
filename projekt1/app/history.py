from collections import Counter;
import time
from gtts import gTTS
import os
import threading

class History:
  MAX_LENGTH = 10
  TIME_BEETWEN_REACTIONS = 5
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
    mytext = f'Siemasz. Kocham programować!'
    t = threading.Thread(target=self._read_text, args=(mytext,))
    t.start()
    
  def _read_text(self, text: str):
    language = 'pl'
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("./main.mp3")
    os.system("afplay main.mp3")


if __name__ == '__main__':
  myHistory = History()
  myHistory.audio_reaction("natural")