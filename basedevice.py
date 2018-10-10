import re
import logging
import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseDevice(ABC):
  ''' Base class for any type of device. Devices are all characterised by having:

      - A fixed length, `len`, which is the number of slots during which the device consumes or
        produces some resource.
      - A shape which is (`N`,`len`), N is 1 for normal devices, but accounts for a device potentially
        being a composite.
      - A list of low/high resource consumption `bounds` of length `N`*`len`.
      - A concave differentiable utility function `u()`, which represents how much value the device
          gets from consuming / producing a given resource allocation (`N`,`len`) at some price.

    This class is more or less a dumb container for the above settings. Sub classes should implement
    (and vary primarily in the implementation of), the utility function.

    Constraints should be convex but this is not currently enforced. Try to maintain:

      - Device is stateless.
      - Device should be considered immutable (the currently available setters are all used on init).
      - Device is serializable and and constructable from the serialization.

    Note Python3 @properties have been used throughout these classes. They mainly serve as very
    verbose and slow way to protect a field, by only defining a getter. Setters are sparingly defined.
  '''

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __iter__(self):
    ''' BaseDevice may or may not be a composite of other devices. An atomic device should just yield
    itself.
    '''
    yield self

  @abstractmethod
  def u(self, s, p):
    ''' Scalar utility for `s` at `p`. `s` should have the same shape as this Device. '''
    pass

  @abstractmethod
  def deriv(self, s, p):
    ''' Derivative of utility for `s` at `p`. `s` should have the same shape as this Device.
    Return value has same shape as `s`.
    '''
    pass

  @abstractmethod
  def hess(self, s, p=0):
    ''' Hessian of utility for `s` at `p` - normally p should fall out. `s` should have the same
    shape as this Device *but* the return value has shape (len, len).
    '''
    pass

  @property
  @abstractmethod
  def id(self):
    pass

  @property
  @abstractmethod
  def shape(self):
    pass

  @property
  @abstractmethod
  def bounds(self):
    pass

  @property
  @abstractmethod
  def lbounds(self):
    pass

  @property
  @abstractmethod
  def hbounds(self):
    pass

  @property
  @abstractmethod
  def constraints(self):
    pass

  @abstractmethod
  def project(self, s):
    ''' project s into cnvx space of this device a return point. Projection should always be possible. '''
    pass

  @abstractmethod
  def to_dict(self):
    pass

  def step(self, s, p, stepsize, solver_options={}):
    ''' Take one step towards optimal demand for price vector `p`, using stepsize plus limited
    minimization. This means stepsize sets the upper bound of change in x, but stepsize can be large
    since limited minimization limits severity of overshoots. This method does not modify the agent.
    @see Device.solve()
    '''
    # Step & Projection
    s_next = s + stepsize * self.deriv(s, p)
    (s_next, o) = project(s_next, s, self.bounds, self.constraints)
    if not o.success:
      if o.status == 8:
        logger.warn(o)
      else:
        raise OptimizationException(o)
    # Limited minimization
    ol = minimize(
      lambda x, p=p: -1*self.u(s + x*(s_next - s), p),
      0.,
      method='SLSQP',
      bounds = [(0, 1)],
      options = solver_options,
    )
    if not ol.success:
      if ol.status == 8:
        logger.warn(ol)
      else:
        raise OptimizationException(ol)
    s_next = s + ol.x*(s_next - s)
    return (s_next, ol)

  def solve(self, p, s0=None, solver_options={}):
    ''' Find the optimal demand for price for the given self and return it. Works on any agent
    since only requires s and self.deriv(). This method does not modify the agent.
    Note AFAIK scipy.optimize only provides two methods that support constraints:
      - COBYLA (Constrained Optimization BY Linear Approximation)
      - SLSQP (Sequential Least SQuares Programming)
    Only SLSQP supports eq constraints. SLSQP is based on the Han-Powell quasi–Newton method. Apparently
    it uses some quadratic approximation, and the same method seems to be sometimes referred called
    SLS-Quadratic-Programming. This does not mean it is limited to quadratics. It should work with *any*
    convex nonlinear function over a convex set.

    SLSQP doesnt take a tol option, only an ftol options. Using this option in the context of this
    software implies tolerance is +/- $ not consumption.

    @see http://www.pyopt.org/reference/optimizers.slsqp.html
    @see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    '''
    _solver_options = {'ftol': 1e-6, 'maxiter': 500}
    _solver_options.update(solver_options)
    logger.debug(_solver_options)

    s0 = s0 if s0 is not None else self.project(np.zeros(self.shape))

    if (self.bounds[:, 0] == self.bounds[:, 1]).all():
      return (self.lbounds, None)
    o = minimize(
      lambda s, p=p: -1*self.u(s, p),
      s0,
      jac=lambda s, p=p: -1*self.deriv(s, p),
      method='SLSQP',
      bounds = self.bounds,
      constraints = self.constraints,
      options = _solver_options,
    )
    if not o.success:
      raise OptimizationException(o)
    return (o.x, o)

  @classmethod
  def from_dict(cls, d):
    ''' Just call constructor. Nothing special to do. '''
    return cls(**d)


def zero_mask(x, fn=None, row=None, col=None, cnt=1):
  if row is None and col is None:
    raise ValueError('row or col argument must be supplied');
  if row is not None and col is not None:
    raise ValueError('row and col arguments are mutually exclusive')
  i = x[row:row+cnt,:] if row is not None else x[:, col:col+cnt]
  o = fn(i).reshape(i.shape)
  r = np.zeros(x.shape)
  if row is not None:
    r[row:row+cnt,:] = o
  else:
    r[:, col:col+cnt] = o
  return r


def project(p, x0, bounds=[], constraints=[], solver_options={}):
  ''' Find the point in feasible region closest to p. '''
  p = p.flatten()
  options = {
    'ftol': 1e-10,
    'disp': False,
    'maxiter': 200
  }
  options.update(solver_options)
  o = minimize(lambda s, p=p: ((s - p)**2).sum(), x0, method='SLSQP',
    jac=lambda s, p=p: 2*(s - p),
    options = options,
    bounds = bounds,
    constraints = constraints
  )
  return (o.x.reshape(x0.shape), o)


class OptimizationException(Exception):  # pragma: no cover
  ''' Some type of optimization error. '''
  o = None # An optional optimization method specific status report (i.e. OptimizeResult).

  def __init__(self, *args):
    self.o = args[0] if args else None
    super(Exception, self).__init__(*args)
