# Projekt 1: Rozpoznawanie emocji osób widzianych przez kamerkę

### Użyte biblioteki:
- tensorflow
- keras
- opencv

### Począkowy treningowy: https://www.kaggle.com/datasets/msambare/fer2013
### Ulepszony zbiór treningowy https://www.kaggle.com/datasets/arnabkumarroy02/ferplus/data

## Informacje o modelu
| Layer (type)          | Output Shape     | Param #    |
|-----------------------|------------------|------------|
| conv2d (Conv2D)       | (None, 46, 46, 32)| 320        |
| conv2d_1 (Conv2D)     | (None, 44, 44, 64)| 18,496     |
| max_pooling2d         | (None, 22, 22, 64)| 0          |
| dropout               | (None, 22, 22, 64)| 0          |
| conv2d_2 (Conv2D)     | (None, 20, 20, 128)| 73,856     |
| max_pooling2d_1       | (None, 10, 10, 128)| 0          |
| conv2d_3 (Conv2D)     | (None, 8, 8, 128)  | 147,584    |
| max_pooling2d_2       | (None, 4, 4, 128)  | 0          |
| dropout_1             | (None, 4, 4, 128)  | 0          |
| flatten               | (None, 2048)       | 0          |
| dense                 | (None, 1024)       | 2,098,176  |
| dropout_2             | (None, 1024)       | 0          |
| dense_1               | (None, 8)          | 8,200      |

Total params: 2,346,632 (8.95 MB)
Trainable params: 2,346,632 (8.95 MB)
Non-trainable params: 0 (0.00 B)

