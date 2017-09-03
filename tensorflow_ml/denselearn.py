import tensorflow as tf 

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("boolean_value",False,"test flags")

if __name__ == "__main__":
	print(flags.FLAGS.boolean_value)