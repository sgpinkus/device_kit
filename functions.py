'''
Various preference functions. Can be used with ADevice.
'''
import numpy as np
from numpy import stack, hstack, zeros, ones, poly1d, polyadd
import numdifftools as nd


class poly2d():
  ''' Wraps up an vector of poly1ds. Makes it slightly easier to get derivs. '''
  _c = None
  _f = None
  _polys = None

  def __init__(self, c):
    ''' `c` should be an array of arrays. Each array the coeffs of a poly1d. '''
    self._c = np.array(c)
    self._polys = [poly1d(c) for c in self._c]
    self._f = lambda x: np.array([self._polys[k](v) for k, v in enumerate(x.reshape(len(self)))])

  def __call__(self, x):
    return self._f(x)

  def __len__(self):
    return len(self._c)

  def deriv(self, n=1):
    return poly2d([polyadd(zeros(len(c)), poly1d(c).deriv(n).coeffs) for c in self._c])

  def hess(self):
    return lambda x: np.diag(self.deriv(2)(x))

  @property
  def coeffs(self):
    return self._c


class InformationEntropy():
  ''' Information entropy preference function. Entropy is bad.
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, c=1):
    self.c = c # Scalar coefficient.

  def __call__(self, r):
    return self.c*InformationEntropy.info_entropy(r)

  def deriv(self):
    return nd.Jacobian(lambda x: self.c*InformationEntropy.info_entropy(x))

  def hess(self):
    return nd.Hessian(lambda x: self.c*InformationEntropy.info_entropy(x))

  @staticmethod
  def info_entropy(r):
    r = np.array([i for i in np.abs(r) if i])
    r = r/r.sum()
    return (r*np.log(r)).sum()


class TemporalVariance():
  ''' Equivalent to moment of inertia if time is distance and consumption is mass. Inertia is bad.
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, c=1):
    self.c = c # Scalar coefficient.

  def __call__(self, r):
    return -self.c*TemporalVariance.inertia(r)

  def deriv(self):
    return nd.Jacobian(lambda x: -self.c*TemporalVariance.inertia(x))

  def hess(self):
    return nd.Hessian(lambda x: -self.c*TemporalVariance.inertia(x))

  @staticmethod
  def inertia(r):
    t = np.arange(len(r))
    return (((t - TemporalVariance.com(r))**2)*r).sum()

  @staticmethod
  def com(r):
    ''' Center of Mass. Merely weighted avg of time-slots. '''
    return np.average(np.arange(len(r)), weights=r)


class CobbDouglas():
  ''' Cobb Douglas *with* constant returns to scale. '''

  def __init__(self, a, c=1):
    self.a = a # Cobb Douglas coefficients.
    self.c = c # Scalar coefficient.

  def __call__(self, r):
    return self.c*CobbDouglas.cobb_douglas(r, self.a)

  def deriv(self):
    return nd.Jacobian(lambda x: self.c*CobbDouglas.cobb_douglas(x, self.a))

  def hess(self):
    return nd.Hessian(lambda x: self.c*CobbDouglas.cobb_douglas(x, self.a))

  @staticmethod
  def cobb_douglas(r, a):
    return (r**(a/a.sum())).prod()
