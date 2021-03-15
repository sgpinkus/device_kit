import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import device_kit
from device_kit import *

def random_uncontrolled():
  return np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1,1, 24)))

def make_model():
  ''' Small power network model. '''
  np.random.seed(19)
  cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
  model = DeviceSet('site1', [
      Device('uncontrolled', 24, (random_uncontrolled(),)),
      IDevice2('scalable', 24, (0.5, 2), (0, 24), **{'d0': 0.3}),
      CDevice('shiftable', 24, (0, 2), (12, 24)),
      GDevice('generator', 24, (-10,0), None, **{'cost': cost}),
      DeviceSet('sub-site1', [
          Device('uncontrolled', 24, (random_uncontrolled(),)),
          SDevice('buffer', 24, (-7, 7), **{ 'c1': 1.0, 'capacity': 70, 'sustainment': 1, 'efficiency': 1})
        ],
        sbounds=(0,10) # max capacity constraint.
      ),
    ],
    sbounds=(0,0) # balanced flow constraint.
  )
  return model

def main():
  model = make_model()
  (x, solve_meta) = device_kit.solve(model, p=0) # Convenience convex solver.
  print(solve_meta.message)
  df = pd.DataFrame.from_dict(dict(model.map(x)), orient='index')
  df.loc['total'] = df.sum()
  pd.set_option('display.float_format', lambda v: '%+0.3f' % (v,),)
  print(df.sort_index())
  print('Utility: ', model.u(x, p=0))
  df.transpose().plot(drawstyle='steps', grid=True)
  plt.ylabel('Power (kWh)')
  plt.xlabel('Time (H)')
  plt.savefig('synopsis.png');

if __name__ == '__main__':
  main()
