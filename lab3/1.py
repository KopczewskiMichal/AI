import pandas as pd
from sklearn.model_selection import train_test_split

def main():
  df = pd.read_csv('iris.csv')
  (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=876987)

  def classify_iris(sl, sw, pl, pw):
    if pw < 0.8:
      return "setosa"
    elif pl > 4.9:
      return "virginica"
    else: 
      return "versicolor"
    
  len = test_set.shape[0]

  train_inputs = train_set[:, 0:4]
  train_classes = train_set[:, 4]
  test_inputs = test_set[:, 0:4]
  test_classes = test_set[:, 4]

  helper(pd.DataFrame(train_set, columns=["sl", "sw", "pl", "pw", "species"]))

  good_predictions = 0
  for i in range(len):
    if classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3]) == test_classes[i]:
      good_predictions += 1

  print("Trafiono:   ",good_predictions)
  print(round(good_predictions/len*100), "%")


def helper(data_set):
  def count_stats(data_set, species):
    df = data_set.loc[data_set["species"] == species]
    print(f"{round(df["sl"].median(), 1)} {round(df["sw"].median(), 1)} {round(df["pl"].median(), 1)} {round(df["pw"].median(), 1)} {species}")
    
    

  for elem in ["setosa", "versicolor", "virginica"]:
    count_stats(data_set, elem)

    

        
          
            





if __name__ == "__main__":
  main()