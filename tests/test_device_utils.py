import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))
import numpy as np
from copy import deepcopy
import unittest
from unittest import TestCase
from device_kit.utils import *
from device_kit.functions import poly2d


class TestBaseDevice(TestCase):

  def test_poly2d_shape(self):
    f = poly2d(np.random.randint(0, 4, (24, 4)))
    r = np.random.random(24)
    self.assertEqual(len(f(r)), 24)
    self.assertEqual(len(f.deriv()(r)), 24)
    self.assertTrue((f(r) == f(r.reshape(1,24))).all())
    self.assertTrue((f(r) == f(r.reshape(24,1))).all())
    self.assertTrue((f.deriv()(r) == f.deriv()(r.reshape(1,24))).all())
    self.assertTrue((f.deriv()(r) == f.deriv()(r.reshape(24,1))).all())

  def test_soc(self):
    r = np.random.random(24)
    v = soc(r, 1, 1)
    # print(v)
    # print(soc(r.reshape(24,1), 1, 1))


if __name__ == '__main__':
    unittest.main()
