import tensorflow as tf 
import math

def ConstructPoint(x,y):
	x1 = tf.constant(x,dtype=tf.float32)
	y1 = tf.constant(y,dtype=tf.float32)
	return tf.stack([x1,y1])

def SolveLinearEquation():
	point1 = ConstructPoint(2,9)
	point2 = ConstructPoint(-1,3)
	X = tf.transpose(tf.stack([point1,point2]))
	print(X)

	B = tf.ones((1,2),dtype=tf.float32)

	parameters = tf.matmul(B, tf.matrix_inverse(X))

	with tf.Session() as session:
		A = session.run(parameters) 

	b = 1/A[0][1]
	a = -b * A[0][0]
	print("Equation: y = {a}x + {b}".format(a=a,b=b))

def SolveCircleEquation():
	points = tf.constant([[2,1],[0,5],[-1,2]],dtype=tf.float64)
	A = tf.constant([
		[2,1,1],
		[0,5,1],
		[-1,2,1]
		],dtype=tf.float64)
	
	B = -tf.constant([[5],[25],[5]],dtype = 'float64')
	X = tf.matrix_solve(A,B)

	with tf.Session() as session:
		result = session.run(X)
		D,E,F = result.flatten()

		print("Equation:x**2 + y**2 + {D}x + {E}y + {F}".format(**locals()))

def ConstrucAinEquation(x,y):
	return[x*x,y*y,x*y,x,y]
def sqrt(x):
	return math.sqrt(x)

def SolveGeneralCircleEquation():

	A = tf.constant([
		ConstrucAinEquation(8,0),
		ConstrucAinEquation(4,-2*sqrt(6)),
		ConstrucAinEquation(-2*sqrt(14),2),
		ConstrucAinEquation(-1*sqrt(46),3),
		ConstrucAinEquation(sqrt(14),5),
		],dtype='float64')

	B = -tf.constant([[-1],[-1],[-1],[-1],[-1]],dtype='float64')

	X = tf.matrix_solve(A,B)

	with tf.Session() as session:
		result = session.run(X)
		A,B,C,D,E = result.flatten()

		print("Equation:{A}x**2 + {B}y**2 + {C}xy + {D}x + {E}y + 1".format(**locals()))


if __name__ =='__main__':
	SolveLinearEquation()
	SolveCircleEquation()
	SolveGeneralCircleEquation()