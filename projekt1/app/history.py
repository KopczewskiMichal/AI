from collections import Counter;
import time
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
  
  def _predict_longer_emotion(self, reaction):
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
      self._predict_longer_emotion(self._print_emotion)
    else:
      print("I did nothing")
    
