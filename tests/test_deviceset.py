import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
import unittest
from unittest import TestCase
from device_kit import *
from device_kit.tests.test_device import test_device, test_cdevice, test_idevice


np.set_printoptions(
  precision=6,
  linewidth=1e6,
  threshold=1e6,
  formatter={
    'float_kind': lambda v: '%0.6f' % (v,),
    'int_kind': lambda v: '%0.6f' % (v,),
  },
)


class TestDeviceSet(TestCase):
  devices = [
    Device(**test_device),
    Device(**test_device),
    CDevice(**test_cdevice),
    IDevice(**test_idevice),
  ]

  def test_basic_properties(self):
    d = DeviceSet('test', self.devices)
    self.assertEqual(len(d), 24)
    self.assertEqual(len(d.id), 4)
    self.assertEqual(d.devices, self.devices)
    self.assertEqual(d.shape, (4, 24))
    self.assertTrue((d.partition[:,1] == 1).all())
    self.assertEqual(d.bounds.shape, (4*24, 2))
    self.assertEqual(d.sbounds, None)
    self.assertEqual(len(d.constraints), 6)

  def test_preferences(self):
    self._test_preferences(DeviceSet('test', self.devices))

  def _test_preferences(self, d):
    x = np.random.random(d.shape)*10
    p = np.random.random(len(d))
    _u = np.array([d.u(x[i], p) for i, d in enumerate(self.devices)]).sum()
    _d = np.concatenate([d.deriv(x[i], p) for i, d in enumerate(self.devices)]).reshape(d.shape)
    self.assertEqual(d.u(x, p), _u)
    self.assertTrue((np.abs(d.deriv(x, p) - _d) < 1e-8).all())
    d.hess(x, p)

  def test_initializer(self):
    d = DeviceSet('test', self.devices, sbounds=(0,100))
    self.assertEqual(len(d.constraints), 6+48)

  def test_set_of_sets(self):
    d1 = DeviceSet('d1', self.devices)
    d2 = DeviceSet('d2', self.devices)
    d = DeviceSet('test', [d1, d2])
    self.assertEqual(d.shape, (8, 24))
    self.assertTrue((d.partition[:,1] == 4).all())
    self.assertEqual(d.bounds.shape, (8*24, 2))

  def test_leaf_devices(self):
    d = DeviceSet('test', self.devices, sbounds=(0,100))
    self.assertEqual(len(list(d.leaf_devices())), 4)

  def test_map(self):
    d1 = DeviceSet('d1', self.devices)
    d2 = DeviceSet('d2', self.devices)
    device = DeviceSet('test', [d1, d2])
    _map = list(device.map(np.ones((8,24))))
    self.assertEqual(len(_map), 8)
    self.assertEqual(_map[0][0], 'test.d1.test')
    self.assertEqual(_map[4][0], 'test.d2.test')
    self.assertEqual(_map[0][1].tolist(), np.ones(24).tolist())

  @unittest.skip('Demonstrates a known bug')
  def test_conflicting_sbounds(self):
    ''' If we set sbounds to <0.5 then the constraints are unsat because that device has a min cbound of
    12=24*0.5. However minimize report no error and returns an infeasible soln, with the sub balancing
    constraint being ignored. But this should occur in DeviceSet too.
    '''
    deviceset = DeviceSet('x', [Device('a', 24, (0,1)), Device('b', 24, (0,1), (12,24))], (0, 0.4))
    soln = solve(deviceset, 0)
    self.assertTrue((np.abs(soln[0][1] - 0.5) <= 1e-6).all()) # Pass
    self.assertNotEqual(soln[1].status, 0) # Fail

class TestMFDeviceSet(TestCase):

  def _test_devices(self, s=23):
    np.random.seed(s)
    uncntrld = 0 # np.maximum(0, np.random.random(24)-0.4)
    cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
    devices = OrderedDict([
        # ('uncntrld', Device('uncntrld', 24, (uncntrld, uncntrld))),
        ('scalable', IDevice2('scalable', 24, (0., 2), (12, 18))),
        ('shiftable', CDevice2('shiftable', 24, (0, 2), (12, 24), a=0.5)), # IDevice2('shiftable', 24, (0, 2), (12, 24) Same same.
        ('generator', GDevice('generator', 24, (-50,0), None, **{'cost': cost})),
    ])
    return devices

  def test_single_device_soln_equivalence(self):
    ''' MFDeviceSet gets flow from N sources. If flows are completely unconstrained shouldn't matter
    if there are N or 1.
    '''
    for k, device in self._test_devices().items():
      mfdevice = MFDeviceSet(device, ['e', 'h'])
      soln1 = solve(device, 0)[0]
      soln2 = solve(mfdevice, 0)[0]
      self.assertTrue(-1e-6 <= device.u(soln1, 0) - mfdevice.u(soln2, 0) <= 1e-6)

  def test_deviceset_soln_equivalence(self):
    ''' Given the set of devices returned by _test_devices(), if we only constrain the generator to
    produce flow in some constant proportion, and all other devices are unconstrained MF devices,
    we sould theoretically get a same (or close to the same) solution as when no MF is present, since
    devices can take from any flow arbitrarily, and should do so such that the generator constraint
    is satisfied.
    '''
    devices = self._test_devices()
    da = DeviceSet('a', list(devices.values()), (0.,0.))
    db = SubBalancedDeviceSet(
      'b',
      [
        # MFDeviceSet(devices['uncntrld'], ['e', 'heat']),
        MFDeviceSet(devices['scalable'], ['e', 'heat']),
        MFDeviceSet(devices['shiftable'], ['e', 'heat']),
        TwoRatioMFDeviceSet(devices['generator'], ['e', 'heat'], [1,8]),
      ],
      sbounds=(0.,0.),
      labels=['heat'],
    )
    (xa, statusa) = solve(da, 0, solver_options={'ftol': 1e-6})
    (xb, statusb) = solve(db, 0, solver_options={'ftol': 1e-6})

    df_xa = pd.DataFrame.from_dict(dict(da.map(xa)), orient='index').transpose()
    df_xa.columns = [i.strip('a.') for i in df_xa.columns]
    df_xb = pd.DataFrame.from_dict(dict(db.map(xb)), orient='index').transpose()
    for k in ['scalable', 'shiftable', 'generator']: # ['uncntrld', 'scalable', 'shiftable', 'generator']:
      df_xb[k] = df_xb.filter(regex='%s' % (k,), axis=1).sum(axis=1)
      del df_xb['b.' + k + '.heat'], df_xb['b.' + k + '.e']
    df_xa.sort_index(axis=1, inplace=True)
    df_xb.sort_index(axis=1, inplace=True)
    self.assertAlmostEqual(da.u(xa, 0), db.u(xb, 0), 5)
    self.assertTrue((np.abs(df_xa.values - df_xb.values) <= 1e-3).all())


class TestSubBalancedDeviceSet(TestCase):

  def _test_devices(self, s=23):
    np.random.seed(s)
    uncntrld = np.minimum(0, np.random.random(24))
    cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
    devices = OrderedDict([
        ('uncntrld', Device('uncntrld', 24, (uncntrld, uncntrld))),
        ('scalable', IDevice2('scalable_bal_me', 24, (0., 2), (12, 18))),
        ('shiftable', CDevice('shiftable', 24, (0, 2), (12, 24), a=0.5)),
        ('generator', GDevice('generator_bal_me', 24, (-50,0), None, **{'cost': cost})),
    ])
    return devices

  def test_basic_sub_balancing(self):
    ''' The *bal_me devices must sum to zero always, while the other device is free, except for the
    sbound. The sbounds effectively sets a new bound on this device.
    '''
    devices = self._test_devices()
    del devices['uncntrld']
    for sbounds in [(-1,1), (-0.5,0.5)]: # (-0.4, 0.4)]:
      device = SubBalancedDeviceSet('x', list(devices.values()), sbounds, labels=['_bal_me'])
      soln = solve(device, 0)[0]
      self.assertTrue((np.abs(soln[1] - np.ones(24)*sbounds[1]) <= 1e-6).all())
      self.assertTrue((np.abs(soln[0] + soln[2]) <= 1e-6).all())


if __name__ == '__main__':
    unittest.main()
