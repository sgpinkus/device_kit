import numpy as np
from powermarket.device import Device


class PVDevice(Device):
  ''' The PVDevice (Generator) models a simple costless generator such as a simple solar panel. It
  produces power (-ve consumption), at zero cost and has a time varying maximum capacity. The client is
  expected to code the max generating capacity of the device at different times of the day via `lbounds`.
  We enforce that hbounds must be zero for this device (can't consume).

  PVDevice does not have any device specific parameters: A simple PV device can be modeled just as
  a specific configuration of the base `Device` class, plus the utility function listed in this
  class. For example for PV device one might set lbounds of `Device` to something like this:

    solar_intensity = np.maximum(0, np.sin(np.linspace(0, np.pi*2, 24)))
    lbounds = -1*np.minimum(max_rate, solar_intensity*efficiency*area)

  '''

  def uv(self, r, p):
    ''' Utility is profit which = revenue since costs are zero. '''
    return -r*p

  @property
  def bounds(self):
    return Device.bounds.fget(self)

  @property
  def cbounds(self):
    return self._cbounds

  @bounds.setter
  def bounds(self, bounds):
    ''' @override bounds setter to ensure hbounds = 0. '''
    if len(bounds) != len(self):
      raise ValueError('bounds has wrong length (%d)' % len(bounds))
    bounds = np.array(bounds)
    lbounds = np.array(bounds[:,0])
    hbounds = np.array(bounds[:,1])
    if not (hbounds == 0).all():
      raise ValueError('hbounds must be all zeros')
    Device.bounds.fset(self, bounds)

  @cbounds.setter
  def cbounds(self, cbounds):
    ''' @override don't allow cbounds
    @todo to allow -ve cbounds.
    '''
    if cbounds == None:
      self._cbounds = None
    else:
      raise ValueError('cbounds not allowed for PVDevice currently')
