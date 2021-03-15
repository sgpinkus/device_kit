import re
import logging
import numbers
import numpy as np
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


class BaseDevice(ABC):
  ''' Base class for any type of device including composite devices. Devices are all characterised
    by having:

      - A fixed length, `len`, which is the number of slots during which the device consumes or
        produces some resource.
      - A shape which is (`N`,`len`), N is always 1 for atomic devices, but accounts for a device
        potentially being a composite.
      - If composite, a list of low/high resource consumption `bounds` of length `N`*`len`.
      - A differentiable utility function `u()`, which represents how much value the device gets
        from consuming / producing a given resource allocation (`N`,`len`) at some price. Note `u()`
        is also used to represent costs of production (cost is -ve utility).

    Device is more or less a dumb container for the above settings. Sub classes should implement
    (and vary primarily in the implementation of), the utility function and possibly additional
    constraints.

    Other notes:

      - This class declares the necessary interfaces to treat a BaseDevice as a composite. Mainly,
       __iter__(), shapes, partition, leaf_devices().
      - Utility functions should be concave (giving convex cost function), and constraints should be
        convex but this is not currently enforced.
      - Device was intended to be and should be treated as immutable but currently this is not enforced.
      - Because Device was intended to be immutable, while a Device represents flow flexibility
        it does not *have* a state  of flow.
      - Devices should be serializable and constructable from the serialization.
      - Python3 @properties have been used throughout these classes. They mainly serve as very
        verbose and slow way to protect a field, by only defining a getter. Setters are sparingly defined.

    @todo rename this class DeviceSpace?
  '''

  @abstractmethod
  def __len__(self):
    pass

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
  def length(self):
    pass

  @property
  @abstractmethod
  def shape(self):
    ''' Return absolute shape of device flow matrix. '''
    pass

  @property
  @abstractmethod
  def shapes(self):
    ''' Array of shapes of sub devices, if any, else [shape]. '''
    pass

  @property
  @abstractmethod
  def partition(self):
    ''' Returns array of (offset, length) tuples for each sub-device's mapping onto this device's
    flow matrix.
    '''
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
    ''' project `s` into cnvx space of this device a return point. Not guaranteed - reasonable effort only.
    Input value may or may not be flattened, return value should have shape of device.
    '''
    pass

  @abstractmethod
  def to_dict(self):
    pass

  def leaf_devices(self):
    ''' Iterate over flat list of (fqid, device) tuples for leaf devices from an input BaseDevice.
    fqid is the id of the leaf device prepended with the dot separated ids of parents. The input device
    may be atomic or a composite. The function distinguishes between them via support for iteration.
    An order list is returned so the key offset indicates the row offset (under the invariant that
    atomic leaf devices have shape (1,N)). Ex {v: k for k, v in enumerate(OrderedDict(x).keys())}
    '''
    def _leaf_devices(device, fqid, s='.'):
      try:
        for sub_device in device:
          for item in _leaf_devices(sub_device, fqid + s + sub_device.id, s):
            yield item
      except:
          yield (fqid, device)
    items = []
    for item in _leaf_devices(self, self.id, '.'):
      items.append(item)
    return items

  def get(self, name):
    ''' Convenience methor to get a single leaf device named 'name'. If greater than one device has
    name an expection is raised. Also see find.
    '''
    return [v for k, v in dict(self.leaf_devices()).items() if k.endswith(name)][0]

  def find(self, regexp):
    return [v for k, v in dict(self.leaf_devices()).items() if re.match(regexp, k)]

  def map(self, s):
    ''' maps rows of flow matrix `s` to identifiers of atomic devices under this device. Returns
    list of tuples.  You can load this into Pandas like pd.DataFrame(dict(device.map(s))).
    Note this implementation assumes that all leaf devices have shape (1,X).
    '''
    s = s.reshape(self.shape)
    for i, d in enumerate(self.leaf_devices()):
      yield (d[0], s[i:i+1,:].reshape(len(self)))

  def validate_bounds(self, bounds):
    ''' Validate Device style bounds specification. Convert to consistent format which is a
    (len(self), 2) shaped ndarray. Input must have length shape (len(self), 2), or len 1 or len 2.
    Presents a conflict when len(self) = 1 or 3. The former input shape takes precedence and if
    len(self) == 1 or 2 you have to use this input format. Support for the latter cases was added for
    convenience and if len(self) is only 1 or 2 that convenience is minor. Note also the input shape is lost.

    If length is 2; the two values are taken to be specifying the lower and upper bound. The values
    must be a scalar or have the same length as this device. If scalar, the value is repeated len(self)
    times.

    If length is 1; the single value in the list must have same length as the device and is used
    for the lower and upper bounds.
    '''
    if not hasattr(bounds, '__len__'):
      raise ValueError('bounds must be a sequence type')
    fixed = False
    try:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.array(bounds).shape == (len(self), 2):
          fixed = True
    except ValueError:
      pass
    if not fixed:
      if len(bounds) == 2:
        bounds = list(bounds)
      elif len(bounds) == 1:
        bounds = [bounds[0], bounds[0]]
      if isinstance(bounds[0], numbers.Number):
        bounds[0] = np.repeat(bounds[0], len(self))
      if isinstance(bounds[1], numbers.Number):
        bounds[1] = np.repeat(bounds[1], len(self))
      if len(bounds[0]) == len(bounds[1]) == len(self):
        bounds = np.stack((bounds[0], bounds[1]), axis=1)
      if len(bounds) != len(self):
        raise ValueError('bounds has wrong length (%d). Require %d' % (len(bounds), len(self)))
    bounds = np.array(bounds)
    lbounds = np.array(bounds[:, 0])
    hbounds = np.array(bounds[:, 1])
    if not np.vectorize(lambda v: v is None)(bounds).all() and not (hbounds - lbounds >= 0).all():
      raise ValueError('max bound must be >= min bound for all min/max bound pairs: %s' % (str(hbounds - lbounds),))
    return bounds

  @classmethod
  def from_dict(cls, d):
    ''' Just call constructor. Nothing special to do. '''
    return cls(**d)
