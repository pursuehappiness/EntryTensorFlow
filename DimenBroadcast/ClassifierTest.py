from sklearn.datasets import load_digits


import tensorflow as tf
import numpy as np


def variable_summaries(var):
	with tf.name_scope("summaries"):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean',mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('stddev',stddev)
		tf.summary.scalar('max',tf.reduce_max(var))
		tf.summary.scalar('min',tf.reduce_min(var))
		tf.summary.histogram('histogram',var)
					


digits = load_digits()

X = digits.data
y = digits.target

n_features = np.shape(X)[1]

n_classes = 1# len(y)

print(n_classes,n_features)

#print(np.shape(X)[1])
#print(y.max())
#print(y.min())

X_train = tf.placeholder("float32",[None,n_features])
y_train = tf.placeholder("float32",[None])

weights_value = tf.Variable(tf.random_normal([n_features,n_classes]))

biase_value = tf.Variable(tf.random_normal([n_classes]))


y_pre = (tf.matmul(X_train,weights_value) + biase_value)

loss_function = tf.reduce_mean(tf.abs((y_train-y_pre)))

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss_function)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		sess.run(train_op,feed_dict = {X_train:X,y_train:y})	
		print("Iteration{}:\tLoss={:.6f}".format(i,sess.run(loss_function,{X_train:X,y_train:y})))



