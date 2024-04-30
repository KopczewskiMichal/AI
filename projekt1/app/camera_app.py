from keras.models import load_model
import cv2
import numpy as np
from history import History;

def main():
  model = load_model('models/emotions-better2.keras', compile=False)
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
  emotions_history = History()

  emotion_dict = {0: 'angry', 1: 'contempt', 2:'disgust', 3:'fear', 4:'happy', 5:'neutral', 6:'sad', 7:'suprise'}
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  cv2.ocl.setUseOpenCL(False)
  cap = cv2.VideoCapture(0)
  while True:
      ret, frame = cap.read()
      frame = cv2.flip(frame, 1)
      if not ret:
          break
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      num_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=4) # minNeighbors pierwotnie 4
      

      for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(emotion_prediction))
        propability = round(emotion_prediction[0][maxindex], 2)
        cv2.putText(frame, f"{emotion_dict[maxindex]} {propability * 100}%", (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if num_faces.size == 4:
          emotions_history.handle_emotion(emotion_dict[maxindex])

      cv2.imshow('Video', cv2.resize(frame, (2400, 1800), interpolation=cv2.INTER_CUBIC))
      
      # "q" key will end loop
      if cv2.waitKey(1) & 0xFF == ord('q'): 
          break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()