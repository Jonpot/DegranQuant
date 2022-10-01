import imageio.v3 as iio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import os
import ast
from keras import metrics

# Update me
# 239, 119, 119
startIndex = 240+120
imageBatchSize = 119

def generateFilenames(batch1size=5,batch2size=5,batch3size=5):
    filenames_plain = []
    for i in range(batch1size+1):
        filenames_plain.append(str(i+1000)+"_out.png")
    for i in range(batch2size+1):
        filenames_plain.append(str(i+2000+59)+"_out.png")
    for i in range(batch3size+1):
        filenames_plain.append(str(i+3000)+"_out.png")
    return(filenames_plain)

os.chdir(r'D:\degranulationData\training_images')

m = open("degranulationCountData.txt","r")
degranList = ast.literal_eval(m.readline())
degranList = degranList[(startIndex):(startIndex + imageBatchSize + 1)]
m.close()


# Update me too
filenames = tf.constant(generateFilenames(batch1size=-1,batch2size=-1,batch3size=imageBatchSize))
labels = tf.constant(degranList)

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3, dtype=tf.uint16)
    image = tf.cast(image_decoded, tf.float32)
    image = image / 65535
    return image, label

dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=(imageBatchSize+1))
dataset = dataset.batch(((imageBatchSize+1)//2) +1)

# step 4: create iterator and final input tensor

(train_images, train_labels), (test_images, test_labels) = dataset
print(train_images)
print(train_labels)
print(test_images)
print(test_labels)
class_names = ['0-2','3-5','6-8','9-11','12-14','15-17','18-20','21-23','24-26','27-29']

#train_images = train_images / 255.0
#test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 3)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#callbacks=[
#        tf.keras.callbacks.LearningRateScheduler(
#            lambda epoch: 1e-3 * 10 ** (epoch / 30)
#        )]

model.fit(train_images, train_labels, epochs=100 
    )

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.Greys)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]),color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

model.save(r'C:\Users\jonat\Downloads\tensorflowTesting\cellModel3')

