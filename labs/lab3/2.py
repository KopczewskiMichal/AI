import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

  save_tree_plot(clf)
  test_clf(clf, test_inputs, test_classes)

  

def test_clf(clf, test_inputs, test_classes):
  correct_predictions = 0
  res = clf.predict(test_inputs)
  print(confusion_matrix(test_classes, res))
  for i in range(len(res)):
    if res[i] == test_classes[i]:
      correct_predictions += 1
  
  print("Correct predictions:    ", correct_predictions)
  print(correct_predictions / len(test_inputs) * 100, "%")

def save_tree_plot(clf):
  import graphviz 
  dot_data = tree.export_graphviz(clf, out_file=None, filled=True) 
  graph = graphviz.Source(dot_data) 
  graph.render("iris") 


if __name__ == "__main__":
  main()

# Wygrałem ja, moje drzewo osiągneło 100% dokładności