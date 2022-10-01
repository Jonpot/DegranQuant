import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
model0 = tf.keras.models.load_model(r'C:\Users\jonat\Downloads\tensorflowTesting\finalModel')
model1 = tf.keras.models.load_model(r'C:\Users\jonat\Downloads\tensorflowTesting\cellModel1')
model3 = tf.keras.models.load_model(r'C:\Users\jonat\Downloads\tensorflowTesting\cellModel3')
class_names = ['0-2','3-5','6-8','9-11','12-14','15-17','18-20','21-23','24-26','27-29']

def plot_image(i, predictions_array, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]),color='blue')

def plot_value_array(i, predictions_array):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')





def _parse_function(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3, dtype=tf.uint16)
    image = tf.cast(image_decoded, tf.float32)
    image = image / 65535
    return image

def runPrediction(dir='ERR'):
  os.chdir(dir)
  outdata = []
  for filename in os.listdir(os.getcwd()):
      try:
          im = _parse_function(filename)
          #im = imageio.imread(filename)
          #im = im/65535.0
          #im = np.dstack([im,im,im])
          im = (np.expand_dims(im,0))
          predictions_single0 = model0.predict(im)
          prediction0 = np.argmax(predictions_single0)
          error0 = np.max(predictions_single0)
          predictions_single1 = model1.predict(im)
          prediction1 = np.argmax(predictions_single1)
          error1 = np.max(predictions_single1)
          predictions_single3 = model3.predict(im)
          prediction3 = np.argmax(predictions_single3)
          error3 = np.max(predictions_single3)
          prediction = np.average([prediction0,prediction1,prediction3])
          error = np.average([error0,error1,error3])
          print(prediction,error)
          #plot_value_array(1, predictions_single[0])
          #_ = plt.xticks(range(10), class_names, rotation=45)
          #plt.show()
          outline = filename+":"+str(prediction)+":"+str(error)
          print(outline)
          outdata.append(outline)
      except:
          continue
  m = open("predictedData2.txt","w")
  m.write(str(outdata))
  m.close

runPrediction(dir=r'D:\degranulationData\cropped_images\150a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\150b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\150c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\150d\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\150e\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\150f\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\150g\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\75a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\75b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\75c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\75d\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\37a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\37b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\37c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\37d\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\18a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\18b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\18c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\18d\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\9a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\9b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\9c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\9d\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\4a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\4b\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\2a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\2b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\2c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\2d\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\1a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\1b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\1c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\1d\processedPNG')

runPrediction(dir=r'D:\degranulationData\cropped_images\0a\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\0b\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\0c\processedPNG')
runPrediction(dir=r'D:\degranulationData\cropped_images\0d\processedPNG')