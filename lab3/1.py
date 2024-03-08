import pandas as pd
from sklearn.model_selection import train_test_split

def main():
  df = pd.read_csv('iris.csv')
  (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=876987)

  def classify_iris(sl, sw, pl, pw):
    if pl  < 0.7:
      return "setosa"
    elif pw > 1.8:
      return "virginica"
    else: 
      return "versicolor"
    
  len = test_set.shape[0]

  train_inputs = train_set[:, 0:4]
  train_classes = train_set[:, 4]
  test_inputs = test_set[:, 0:4]
  test_classes = test_set[:, 4]

  # helper(train_set)

  good_predictions = 0
  for i in range(len):
    if classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3]) == test_classes[i]:
      good_predictions += 1

  print(good_predictions)
  print(round(good_predictions/len*100), "%")


def helper(data_set):
  def count_stats(data_set):
    # schemat: sl, sw, pl, pw, count
    setosa_stats = [0,0,0,0,0]
    versicolor_stats = [0,0,0,0,0]   
    virginica_stats = [0,0,0,0,0]

    for elem in data_set:
      match elem[4]:
        case "setosa":
          for i in range(4):
            setosa_stats[i] += elem[i]
          setosa_stats[4] += 1
        case "versicolor":
          for i in range(4):
            versicolor_stats[i] += elem[i]
          versicolor_stats[4] += 1
        case "virginica":
          for i in range(4):
            virginica_stats[i] += elem[i]
          virginica_stats[4] += 1
    
    for i in range(4):
      setosa_stats[i] = round(setosa_stats[i] / setosa_stats[4],2)
      versicolor_stats[i] = round(versicolor_stats[i] / versicolor_stats[4],2)
      virginica_stats[i] = round(virginica_stats[i] / virginica_stats[4],2)
    
    print("setosa:     ", setosa_stats)
    print("versicolor: ", versicolor_stats)
    print("virginica:  ", virginica_stats)

  count_stats(data_set)

    

        
          
            





if __name__ == "__main__":
  main()