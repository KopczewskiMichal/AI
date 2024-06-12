import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from random import randint

def main():
  df = pd.read_csv("diabetes 1.csv")

  network(df, 6, 3)



def network(data, *hidden_layers):
  train_inputs, train_classes, test_inputs, test_classes = split_data(data)
  mlp = MLPClassifier(
                    activation="relu",
                    hidden_layer_sizes=hidden_layers,
                    # hidden_layer_sizes=(6,5),
                    max_iter=500)
  mlp.fit(train_inputs, train_classes)

  # precision_score(y_true, y_pred, zero_division=0)
  pred = mlp.predict(test_inputs)
  print(metrics.classification_report(test_classes, pred, zero_division=0))
  correct_predictions = metrics.accuracy_score(test_classes, pred)
  print(confusion_matrix(test_classes, pred))
  print(f"Skuteczność: {round(correct_predictions*100, 1)}%\nUkryte warstwy: {hidden_layers}\n")
  # print(f"Ukryte warstwy: {hidden_layers}\n")


def split_data(df):
  # random_state = randint(0,1000)
  (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=54233)
  train_inputs = train_set[:, 0:8]
  train_classes = train_set[:, 8]
  test_inputs = test_set[:, 0:8]
  test_classes = test_set[:, 8]
  return train_inputs, train_classes, test_inputs, test_classes

if __name__ == '__main__':
  main()


# Gorsze są błędy FP a niestety jest ich procentowo więcej