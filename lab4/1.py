import math;

def main():
  arr = [
    [23, 75, 176,True],
    [25, 67, 180,True],
    [28, 120, 175,False],
    [22, 65, 165, True],
    [46, 70, 187, True],
    [50, 68, 180, False],
    [48, 97, 178, False],
    [20, 76, 188, True] # Rekord na podstawie danych kolegi, rzeczywiście chciał grać
  ]

  print(test_network(arr))
  

def classify(age, weight, height)->bool:
  def activation(x) -> float:
    return 1/(1 + math.e ** - x)

  first_node = activation(age * -0.46122 + weight * 0.97314 + height * -0.39203 + 0.80109)
  second_node = activation(age * 0.78548 + weight * 2.10584 + height * - 0.57847 + 0.43529)
  third_node = activation(first_node * -0.81546 + second_node * 1.03775 - 0.2368)
  return round(third_node, 0)

def test_network(data):
  correct_count = 0
  for elem in data:
    if classify(elem[0], elem[1], elem[2]) == elem[3]:
      correct_count += 1
      print(elem, "Działa")
    else:
      print(elem, "Nie Działa")
    
  
  return correct_count/len(data) 


if __name__ == "__main__":
  main()