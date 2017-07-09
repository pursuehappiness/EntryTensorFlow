import matplotlib.image as mpimg

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.png"

image = mpimg.imread(filename)

print(image.shape)

import matplotlib.pyplot as plt 
plt.imshow(image)
plt.show()

##################
import tensorflow as tf 

x = tf.Variable(image,name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	x = tf.transpose(x,perm=[1,0,2])
	session.run(model)
	result = session.run(x)

plt.imshow(result)
plt.show()
