import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


def main():
  df = pd.read_csv('iris.csv')



  test_clf(df)
  test_knn(3, df)
  test_knn(5, df)
  test_knn(11, df)

def test_split(df):
  (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=876987)
  train_inputs = train_set[:, 0:4]
  train_classes = train_set[:, 4]
  test_inputs = test_set[:, 0:4]
  test_classes = test_set[:, 4]
  return train_inputs, train_classes, test_inputs, test_classes

def test_knn(n, data_set):
  train_inputs, train_classes, test_inputs, test_classes = test_split(data_set)
  knn = KNeighborsClassifier(n_neighbors=n).fit(train_inputs, train_classes)
  pred = knn.predict(test_inputs)
  correct_predictions = metrics.accuracy_score(test_classes, pred)
  print(f"Wydruk dla {n} sąsiadów")
  print(metrics.confusion_matrix(test_classes, pred))
  print("Correct predictions:    ", round(correct_predictions, 3), "\n")


# Drzewo decyzyjne
def test_clf(df):
  train_inputs, train_classes, test_inputs, test_classes = test_split(df)
  clf = tree.DecisionTreeClassifier()
  clf.fit(train_inputs, train_classes)
  correct_predictions = 0
  pred = clf.predict(test_inputs)
  correct_predictions = metrics.accuracy_score(test_classes, pred)
  print(f"Wydruk dla drzewa decyzyjnego")
  print(metrics.confusion_matrix(test_classes, pred))
  print("Correct predictions:    ", round(correct_predictions, 3), "\n")

  



if __name__ == '__main__':
  main()