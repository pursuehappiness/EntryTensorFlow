import tensorflow as tf 


def VariableTest():
	print('tensorflow variable initialize demo')
	
	x = tf.constant([35,34,20],name = 'x',dtype = tf.float64)
	y = tf.Variable(x+x/2,name = 'y',dtype = tf.float64)

	print(x)
	print(y)

	model = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(model)
		print(sess.run(x))
		print(sess.run(y))


if __name__ == '__main__':
	VariableTest()