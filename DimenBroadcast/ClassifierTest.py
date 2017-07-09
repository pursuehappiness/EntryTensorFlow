from sklearn.datasets import load_digits


import tensorflow as tf
import numpy as np

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

weights_value = tf.Variable(tf.zeros([n_features,n_classes]))

biase_value = tf.Variable(tf.zeros([n_classes]))


y_pre = tf.nn.softmax(tf.matmul(X_train,weights_value) + biase_value)

loss_function = tf.reduce_mean(tf.abs((y_train-y_pre)/y_train))

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)





with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10000):
		sess.run(train_op,feed_dict = {X_train:X,y_train:y})
		if i%100 == 0 :
			print("Iteration{}:\tLoss={:.6f}".format(i,sess.run(loss_function,{X_train:X,y_train:y})))



