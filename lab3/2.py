import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

def main():
  df = pd.read_csv('iris.csv')
  (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=876987)

  train_inputs = train_set[:, 0:4].tolist()
  train_classes = train_set[:, 4].tolist()
  test_inputs = test_set[:, 0:4]
  test_classes = test_set[:, 4]

  clf = tree.DecisionTreeClassifier()
  clf.fit(train_inputs, train_classes)
  tree.plot_tree(clf)

  test_clf(clf, test_inputs, test_classes)

  

def test_clf(clf, test_inputs, test_classes):
  correct_predictions = 0
  res = clf.predict(test_inputs)
  for i in range(len(res)):
    if res[i] == test_classes[i]:
      correct_predictions += 1
  
  print("Correct predictions:    ", correct_predictions)
  print(correct_predictions / len(test_inputs) * 100, "%")


if __name__ == "__main__":
  main()

# Wygrałem ja, moje drzewo osiągneło 100% dokładności