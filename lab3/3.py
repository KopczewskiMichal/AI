import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



def main():
  df = pd.read_csv('iris.csv')



  classify_clf(df)
  classify_knn(3, df)
  classify_knn(5, df)
  classify_knn(11, df)
  classify_NB(df)

def split_data(df):
  (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=876987)
  train_inputs = train_set[:, 0:4]
  train_classes = train_set[:, 4]
  test_inputs = test_set[:, 0:4]
  test_classes = test_set[:, 4]
  return train_inputs, train_classes, test_inputs, test_classes

def classify_knn(n, data_set):
  train_inputs, train_classes, test_inputs, test_classes = split_data(data_set)
  knn = KNeighborsClassifier(n_neighbors=n).fit(train_inputs, train_classes)
  pred = knn.predict(test_inputs)
  correct_predictions = metrics.accuracy_score(test_classes, pred)
  print(f"Wydruk dla {n} sąsiadów")
  print(metrics.confusion_matrix(test_classes, pred))
  print("Correct predictions:    ", round(correct_predictions, 3), "\n")


# Drzewo decyzyjne
def classify_clf(df):
  train_inputs, train_classes, test_inputs, test_classes = split_data(df)
  clf = tree.DecisionTreeClassifier()
  clf.fit(train_inputs, train_classes)
  correct_predictions = 0
  pred = clf.predict(test_inputs)
  correct_predictions = metrics.accuracy_score(test_classes, pred)
  print(f"Wydruk dla drzewa decyzyjnego")
  print(metrics.confusion_matrix(test_classes, pred))
  print("Correct predictions:    ", round(correct_predictions, 3), "\n")

def classify_NB(df):
  train_inputs, train_classes, test_inputs, test_classes = split_data(df)
  nb = GaussianNB().fit(train_inputs, train_classes)
  pred = nb.predict(test_inputs)
  correct_predictions = metrics.accuracy_score(test_classes, pred)
  print(f"Wydruk dla Naive Bayes")
  print(metrics.confusion_matrix(test_classes, pred))
  print("Correct predictions:    ", round(correct_predictions, 3), "\n")

  



if __name__ == '__main__':
  main()