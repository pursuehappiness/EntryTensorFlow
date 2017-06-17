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

	def test_const_add_2dimarray_1dimarray_in_row(self):
		array_2dim = [[1,2,3],[4,5,6]]
		array_1dim = [101,102,103]

		ret = TensorFlowConstAddOneDimeWithTwoDime(array_2dim,array_1dim)

		for i in range(2):
			for j in range(3):
				array_2dim[i][j] += array_1dim[j]
				self.assertEqual(array_2dim[i][j],ret[i][j])

	def test_const_add_2dimarray_1dimarray_in_col(self):
		array_2dim = [[1,2,3],[4,5,6]]
		array_1dim = [[101],[103]]

		ret = TensorFlowConstAddOneDimeWithTwoDime(array_2dim,array_1dim)

		for i in range(2):
			for j in range(3):
				array_2dim[i][j] += array_1dim[i][0]
				self.assertEqual(array_2dim[i][j],ret[i][j],'assert [%d][%d]'% (i,j,))

	def test_const_add_2dim_array(self):
		array_2dim1 = [[1,2,3],[4,5,6]]
		array_2dim2 = [[11,12,13],[24,52,62]]
		
		ret = TensorFlowConstAddTwoObj(array_2dim1,array_2dim2)

		for i in range(2):
			for j in range(3):
				value_ij = array_2dim1[i][j]+array_2dim2[i][j]
				self.assertEqual(value_ij,ret[i][j])


if __name__ == '__main__':
	unittest.main(warnings='ignore')