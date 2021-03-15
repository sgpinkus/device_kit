import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))
from unittest import TestCase
from device_kit.projection import *
import numpy as np


class TestBaseDevice(TestCase):
  ''' Test stuff. '''
  tol = 1e-9
  test_cube = [[0, 1], [-1, 1], [2, 3]]
  test_points = [[0, 0], [1, 1], [0, 1], [1, 0], [-1, -1], [2, 0], [3, 0], [-3, -3], [-67, 89]]
  test_halves = [[1, 1, 1], [1, 2, 1], [-1, -1, 1], [23, 67, -34]]

  def test_hypercube(self):
    cube = HyperCube(self.test_cube)
    for p in [np.random.rand(len(self.test_cube))*np.random.choice([1, 2, 3, 4]) for i in range(1, 10)]:
      pass
    self.assertTrue(cube.is_in(np.array(self.test_cube)[:, 0]))
    self.assertTrue(cube.is_in(np.array(self.test_cube)[:, 1]))
    self.assertTrue(cube.is_in(np.array(self.test_cube)[:, 0]-0.99*cube.tol))
    self.assertTrue(cube.is_in(np.array(self.test_cube)[:, 1]+0.99*cube.tol))
    self.assertTrue(cube.is_in(np.array(self.test_cube)[:, 0]+cube.tol))
    self.assertTrue(cube.is_in(np.array(self.test_cube)[:, 1]-cube.tol))
    self.assertFalse(cube.is_in(np.array(self.test_cube)[:, 0]-1.1*cube.tol))
    self.assertFalse(cube.is_in(np.array(self.test_cube)[:, 1]+1.1*cube.tol))
    self.assertFalse(cube.is_in([100, 100, 100]))

  def test_halfspace(self):
    ''' Test halfspace by doing projection and confirming projected point is in the 1/2 space. '''
    for sign in [-1, 1]:
      for h in self.test_halves:
        h = HalfSpace(h[0:2], h[2], sign)
        for p in map(lambda p: np.array(p), self.test_points):
          proj = h.project(p)
          projdot = proj.dot(h._normal)
          if sign < 0:
            self.assertTrue(projdot <= h._offset+self.tol)
          else:
            self.assertTrue(projdot >= h._offset-self.tol)
    h = HalfSpace(self.test_halves[0][0:2], self.test_halves[0][2], -1)
    self.assertTrue(h.is_in([0.5, 0.5]))
    self.assertTrue(h.is_in([1, 0]))
    self.assertTrue(h.is_in([0, 1]))
    self.assertTrue(h.is_in([1+h.tol, 0]))
    self.assertTrue(h.is_in([1-h.tol, 0]))
    self.assertFalse(h.is_in([1+h.tol*10, 0]))
    self.assertFalse(h.is_in([1, h.tol*10]))

  def test_slice(self):
    pass

  def test_intersection(self):
    pass

  def test_list(self):
    s1 = Slice([1, 1], 11, 12)
    s2 = Slice([-1, -1], 10, 11)
    s3 = Slice([3, 2], -6, 6)
    l = List([s1, s2, s3])
    p = np.random.rand(3, 2)
    self.assertEqual(l._axis, 0)
    self.assertEqual(l._shape, (3, 2))
    self.assertTrue(not l.is_in(p))
    pr = l.project(p)
    self.assertTrue(l.is_in(pr))
    l = List([s1, s2, s3], axis=1)
    self.assertEqual(l._axis, 1)
    self.assertEqual(l._shape, (2, 3))
    p = np.random.rand(2, 3)
    self.assertTrue(not l.is_in(p))
    pr = l.project(p)
    self.assertTrue(l.is_in(pr))

  def test_list_exceptions(self):
    s1 = Slice([1, 1], 11, 12)
    s2 = Slice([-1, -1], 10, 11)
    s3 = Slice([3, 2], -6, 6)
    l = List([s1, s2, s3])
    with self.assertRaises(ValueError):
      l.project(np.random.rand(2, 3))
    l = List([s1, s2, s3], axis=1)
    with self.assertRaises(ValueError):
      l.project(np.random.rand(3, 2))
    with self.assertRaises(ValueError):
      l.project(np.random.rand(3, 3))
    with self.assertRaises(ValueError):
      l.project(np.random.rand(2, 2))
