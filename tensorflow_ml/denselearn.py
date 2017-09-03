import tensorflow as tf 

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("boolean_value",False,"test flags")


def tflearn():
	#first step parser input arguments


	#then loader the saved model


	#after that build a model


	#loade train data

	#loade test data

	#train the model

	#validate the trained model accuracy

	#export the model

if __name__ == "__main__":
	print(flags.FLAGS.boolean_value)