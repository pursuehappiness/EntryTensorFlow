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
	with tf.Session() as session:
		return session.run(add_op)

def TensorFlowConstAddTwoObj(obj1,obj2):
	a = tf.constant(obj1, name = 'a')
	b = tf.constant(obj2, name = 'b')
	print(a.shape)
	print(b.shape)
	add_op = a + b
	with tf.Session() as session:
		return session.run(add_op)

def TensorFlowConstAddOneDimeWithTwoDime(Array_2Dim,Array_1Dim):
	a = tf.constant(Array_2Dim, name = 'a')
	b = tf.constant(Array_1Dim, name = 'b')

	add_op = a + b
	with tf.Session() as session:
		return session.run(add_op)

def TensorFlowConstShape(value):
	return tf.shape(2,3,value)


if __name__ == '__main__':
	print(TensorFlowConstAddTwoObj([[1,2,3],[4,5,6]],[100,101,102]))
	print(TensorFlowConstAddTwoObj([[1,2,3],[4,5,6]],[[100],[101]]))
	print(TensorFlowConstShape(12))

