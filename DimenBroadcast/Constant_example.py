import tensorflow as tf 

def TensorFlowConstant():

	a = tf.constant(3,name='a')

	with tf.Session() as session:
		return (session.run(a))

def TensorFlowConstantAdd(valuea,valueb):
	a=tf.constant(valuea,name = 'a')
	b=tf.constant(valueb,name = 'b')
	add_op = a + b
	with tf.Session() as session:
		return session.run(add_op)

def TensorFlowConstantAddList(list1,list2):
	a = tf.constant(list1, name = 'a')
	b = tf.constant(list2, name = 'b')

	add_op = a + b

	with tf.Session() as session:
		return session.run(add_op)

def TensorFlowConstantAddListwithNum(list1,num):
	a = tf.constant(list1, name = 'a')
	b = tf.constant(num, name = 'b')

	add_op = a + b
	with tf.Session as session:
		return session.run(add_op)