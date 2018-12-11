import numpy as np
import json
from numpy.random import rand
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
from powermarket.network import Network
from powermarket.agent import DeviceAgent
from powermarket.device import *
from powermarket.tests.test_device import *

np_printoptions = {
  'linewidth': 1e6,
  'threshold': 1e6,
  'formatter': {
    'float_kind': lambda v: '%+0.4f' % (v,),
    'int_kind': lambda v: '%+0.4f' % (v,),
  },
}
np.set_printoptions(**np_printoptions)


class TestDevice():
  ''' Basic tests of base device class. '''
  @classmethod
  def test_basics(cls):
    cbounds = None
    a = Device("foo", 24, np.stack((mins, maxs), axis=1), cbounds, None)
    b = Device(*test_device)
    print(a)
    print(b)

  @classmethod
  def test_constraints(cls):
    d = SDevice(*test_sdevice)
    pprint(d.constraints)
    s = np.random.rand(len(d))
    for c in d.constraints:
      print(c['fun'](s))
      print(c['jac'](s))

  @classmethod
  def test_init_clone(cls):
    a = IDevice("foo", 24, np.stack((mins, maxs), axis=1))
    print(a, a.to_dict())
    d = a.to_dict()
    b = IDevice(**d)
    print(b, b.to_dict())


class TestIDevice():
  ''' Informal sanity check, graphing random IDevice utility functions and their derivatives. '''

  def test_all(self):
    self.test_utility_demo()

  def test_utility_demo(self):
    params = {
      'a': 0.1,
      'b': 2,
      'c': 1,
      'd': 0,
    }
    d = IDevice(
      'idevice',
      24,
      np.stack((np.zeros(24), np.ones(24)), axis=1),
      None,
      params,
    )
    ax = np.linspace(-0.5, 1.5)
    plt.plot(ax, np.vectorize(lambda x: d.u(x, np.zeros(24)))(ax))
    plt.axvline(0, label='min', color='k')
    plt.axvline(1, label='max', color='k')
    plt.axvline(1+params['a'], label='a', color='orange')
    plt.axhline(params['d'], label='d', color='red')
    plt.title(json.dumps(params))
    plt.legend()
    plt.grid(True)
    plt.show()

  def test_utility(self):
    num = 6
    mins = rand(num)
    maxs = mins + rand(num)+0.5
    params = {
      'a': rand(num)*0.5,
      'b': 3,
      'c': 1,
      'd': 0
    }
    a = IDevice("test_idevice", num, np.stack((mins, maxs), axis=1), None, params)
    print(a, '---')
    s = (maxs - mins)/20
    v = np.array([[mins + i*s, a.uv(mins + i*s, np.zeros(len(a)))] for i in range(0, 21)])
    print(maxs-mins)
    for i in range(0, num):
      plt.plot(v[:,:, i][:, 0], v[:,:, i][:, 1])
    plt.gca().set_color_cycle(None)
    for i in mins:
      plt.axvline(i, color='k')
    for i in maxs:
      plt.axvline(i, color='r')
    plt.xlim(0, 3)
    plt.ylim(-1, 1.1)
    plt.show()

  def test_utility_deriv(self):
    colors = ['red', 'black', 'lime', 'navy', 'fuchsia', 'green', 'yellow', 'blue', 'orange', 'purple']
    num = 3
    mins = rand(num)
    maxs = mins + rand(num)+0.5
    params = {
      'a': rand(num)*0.5,
      'b': 3,
      'c': 1,
      'd': 0
    }
    a = IDevice("test_idevice", num, np.stack((mins, maxs), axis=1), None, params)
    print(a, '---')
    s = (maxs - mins)/20
    v = np.array([[mins + i*s, a.uv(mins + i*s, np.zeros(len(a)))] for i in range(0, 20)])
    d = np.array([[mins + i*s, a.deriv(mins + i*s, np.zeros(len(a)))] for i in range(0, 20)])
    for i in range(0, num):
      plt.plot(v[:,:, i][:, 0], v[:,:, i][:, 1], '--', color=colors[i%len(colors)])
      plt.plot(d[:,:, i][:, 0], d[:,:, i][:, 1], '-', color=colors[i%len(colors)])
    plt.xlim(0, 3)
    plt.show()


class TestCDevice():
  ''' Test CDevice basics. '''
  @classmethod
  def test_basics(cls):
    a = CDevice(*test_cdevice_2)
    print(a)
    print(a.u(np.zeros(len(a)), np.zeros(len(a))), a.deriv(np.zeros(len(a)), np.zeros(len(a))))
    print(a.hbounds.sum())


class TestGDevice():
  ''' Test GDevice basics. '''

  test_costs = [
    [1, 0, 0],
    [0.045,  0.058, 0.24, 0],
    np.concatenate((np.ones((12, 4))*[0.00045,  0.0058, 0.024, 0], np.ones((12, 4))*[0.73,  0.58, 0.024, 1])),
  ]

  @classmethod
  def test_basics(cls):
    d = cls.get_test_device()
    print(d)
    print(d.bounds, d.lbounds, d.hbounds)
    print(d.deriv(-2*np.ones(len(d)), np.ones(len(d))))
    print(d.deriv(-1*np.ones(len(d)), np.ones(len(d))))
    print(d.deriv(-0*np.ones(len(d)), np.ones(len(d))))
    print(d._cost_function(2*np.ones(len(d))))

  @classmethod
  def test_device_agent(cls):
    d = DeviceAgent(cls.get_test_device(1))
    for i in np.linspace(0, 5, 21):
      d.solve(p=ones*i)
      print('p', d.p)
      print('r', d.r)
      print('d', d.deriv(d.r, d.p))
      print('---')

  @classmethod
  def test_plot(cls):
    x = np.linspace(0, -5, 100)
    d = DeviceAgent(cls.get_test_device())
    plt.plot(x, np.vectorize(lambda r: d.u(r*ones, ones))(x), label='0')
    d = DeviceAgent(cls.get_test_device(1))
    plt.plot(x, np.vectorize(lambda r: d.u(r*ones, ones))(x), label='1')
    d = DeviceAgent(cls.get_test_device(2))
    plt.plot(x, np.vectorize(lambda r: d.u(r*ones, ones))(x), label='2')
    plt.legend()
    plt.ylim(-100, None)
    plt.show()

  @classmethod
  def get_test_device(cls, i=0):
    return GDevice('test', 24, np.stack((-10*ones, zeros), axis=1), None, {'cost': cls.test_costs[i]})


class TestPVDevice():
  ''' Test PVDevice basics. '''
  @classmethod
  def test_basics(cls):
    d = cls.get_test_device()
    print(d)
    print(d.bounds, d.lbounds, d.hbounds)

  @classmethod
  def test_device_agent(cls):
    d = DeviceAgent(cls.get_test_device())
    print(d)
    d.update(np.zeros(len(d)))
    print(d)
    d.update(np.ones(len(d)))
    print(d)

  @classmethod
  def get_test_device(cls):
    max_rate = 2
    area = 2.5
    efficiency = 0.9
    solar_intensity = np.maximum(0, np.sin(np.linspace(0, np.pi*2, 24)))
    lbounds = -1*np.minimum(max_rate, solar_intensity*efficiency*area)
    return PVDevice('solar', 24, np.stack((lbounds, np.zeros(24)), axis=1), None, None)

class TestSDevice():
  ''' Test SDevice basics. '''

  def test_all(self):
    self.test_basics()
    self.test_sdevice()
    self.test_costs_at_zero_are_zero()
    self.test_cd_solution()
    self.test_charge_at()

  def test_sdevice(self):
    self.test_basics()
    self.test_costs_at_zero_are_zero()
    self.test_cd_solution()

  def test_basics(self):
    a = self.get_test_device()
    print(a, '\n', a.u(), '\n', a.deriv())
    p = np.random.rand(24)
    a.update(p)
    print(a)
    plt.plot(p, label='price')
    plt.plot(a.r, label='consumption')
    plt.legend()
    plt.show()

  def test_costs_at_zero_are_zero(self):
    ''' Cost should be zero when charge is zero. and only increase. '''
    a = self.get_test_device()
    ax = np.linspace(-1, 1, 50)
    plt.plot(ax, np.vectorize(lambda x: a.u(s=x*np.ones(len(a))))(ax))
    plt.show()

  def test_cd_solution(self):
    a = DeviceAgent(
      SDevice(
        'battery',
        24,
        np.stack((np.ones(24)*-5, np.ones(24)*5), axis=1),
        None,
        {'c1': 1, 'c2': 0.5, 'c3': 0.2, 'capacity': 5}
      )
    )
    print(a)
    p = np.cos(np.linspace(0, 2*np.pi, 24))*5+5
    a.update(p)
    print('Charge:')
    print(['%.3f' % (c,) for c in a.charge_at(a.r)])
    print('Damage:')
    print(a.deep_damage_at(a.r))
    plt.axhline(0)
    plt.plot(p, label='price')
    plt.plot(a.r, label='consumption')
    plt.xlim(0, 23)
    plt.legend()
    plt.show()

  def test_charge_at(self):
    sustainments = [1, 0.99, 0.98, 0.95, 0.90, 0.80]
    d = SDevice(*test_sdevice)
    r1 = np.concatenate(([1], np.zeros(23)))
    r2 = r1.copy()
    r2[14] = 1
    r3 = r2.copy()
    r3[5] = -1
    b = np.concatenate(([d.base()], np.zeros(24)))
    ax = np.arange(0, 24)
    ax1 = np.arange(-1, 24)
    colors = ['blue', 'red', 'orange', 'yellow', 'purple', 'fuchsia', 'lime', 'green']
    for r in [r1, r2, r3]:
      for i, s in enumerate(sustainments):
        d.sustainment = s
        plt.plot(d.charge_at(r), colors[i], label=s)
      plt.bar(ax1, b, 1, color='b')
      plt.bar(ax, r, 1, color='b')
      plt.legend()
      plt.xlabel('Time')
      plt.ylabel('RoC (bars) and SoC (lines)')
      plt.grid(True, zorder=5)
      plt.show()

  def get_test_device(self):
    return DeviceAgent(
      SDevice(
        'battery',
        24,
        np.stack((np.ones(24)*-1, np.ones(24)), axis=1),
        None,
        {'c1': 0.1, 'c2': 0.05, 'c3': 0.001}
      )
    )

TestSDevice().test_all()
# TestIDevice().test_all()