from pprint import pprint
from einops import rearrange
import numpy as np

np.random.seed(0)
x = np.random.randint(0, 10, (3, 3, 3))
y = rearrange(x, 'h w c -> c (h w)')
pprint (x)
pprint (y)