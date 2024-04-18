import cv2

def main(imagePath: str) -> None:
  image = cv2.imread(imagePath)
  print(image.shape[:3])
  create_better_bw(image)
  # create_simple_bw(image)


def create_simple_bw(image) -> None:
  (row, col) = image.shape[0:2]
  for i in range(row):
    for j in range(col):
      image[i, j] = sum(image[i, j]) * 0.33
    # Displaying the image
  cv2.imwrite("2_images/simple.jpg", image) 

def create_better_bw(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imwrite("2_images/better.jpg", image) 


if __name__ == "__main__":
  main("2_images/orig.jpeg")