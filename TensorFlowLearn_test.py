import unittest
from DimenBroadcast.Constant_example import *

class TensorFlowTestCase(unittest.TestCase):

	def test_const(self):
		ret = TensorFlowConstant()
		self.assertEqual(ret,3)

	def test_const_add(self):
		ret = TensorFlowConstantAdd(4,5)
		self.assertEqual(4+5,ret)

		ret = TensorFlowConstantAdd(0,0)
		self.assertEqual(0,ret)

	def test_const_add_list(self):
		list1 = [1,2,3]
		list2 = [4,2,6]
		ret = TensorFlowConstantAddList(list1,list2)
		self.assertEqual([1,2,3,4,2,6],list1+list2)
		list3=[5,4,9]
		for dati in range(3):
			self.assertEqual(list3[dati],ret[dati])

	def test_const_add_list_with_num(self):
		list1 = [13,6,9]
		num=3
		ret = TensorFlowConstantAddList(list1,num)
		list3=[16,9,12]
		#self.assertEqual(list1+num,list3)
		for dati in range(3):
			self.assertEqual(list3[dati],ret[dati])


if __name__ == '__main__':
	unittest.main(warnings='ignore')