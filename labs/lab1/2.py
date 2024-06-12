import math
from random import randint
import matplotlib.pyplot as plt
import numpy as np
v_0 = 50 # m/s
h_0 = 100 # m
possible_target = [50, 340] # m
precision = 5 # m
g = 9.81 # m/s^2

def distance(angle) -> int:
  angle_radians = math.radians(angle)
  sin_angle = math.sin(angle_radians)
  distance = (v_0 * sin_angle + math.sqrt(v_0 ** 2 * sin_angle ** 2 + 2 * g * h_0)) * v_0 * math.cos(angle_radians) / g
  return round(distance)

def mk_plot(angle):
  plt.style.use('_mpl-gallery')
  x = np.linspace(0, 350)
  angle_radians = np.radians(angle)
  y = (-g / (2 * v_0**2 * math.cos(angle_radians)**2)) * x**2 + (math.sin(angle_radians) / math.cos(angle_radians)) * x + h_0

  plt.figure(figsize=(12, 7)) 
  plt.plot(x, y)  

  plt.title("Trajektoria pocisku")
  plt.xlabel("Odległość (m)")
  plt.ylabel("Wysokość (m)")
  plt.xlim(0, 350)  
  plt.ylim(0, 300) 
  plt.grid(True) 
  plt.tight_layout()
  plt.savefig('plot.png')
  plt.show()



def main():
  target = randint(possible_target[0], possible_target[1])

  print(f"Cel znajduję się w odległości {target}m\nTolerancja błędu wynosi {precision}m")
  shots_counter = 1

  while True:
    angle = input("Wprowadź kąt w stopniach: ")
    if angle.isdigit() == False:
      print("Kąt musi być liczbą całkowitą")
    elif int(angle) < 0 or 90 < int(angle):
      print("Kąt musi być ostry")
    else:
      angle = int(angle)
      dist = distance(angle)
      if dist in range(target - precision, target + precision + 1):
        print(f"Gratulacje, udało się trafić w cel za {shots_counter} razem!!!\nZakońćzono grę")
        mk_plot(angle)
        break
      else:
        print(f"Niestety nie udało się trafić w cel, pocisk wylądował w odległości {dist}")
        shots_counter += 1
      

if __name__ == "__main__":
  main()

# Czas pracy 2 godziny
  