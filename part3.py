import torch
import numpy as np
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normal view
X, Y = np.mgrid[-0.1:0.1:0.0001, -0.1:0.1:0.0001]
# zoomed in
#X, Y = np.mgrid[-0.001:0.01:0.00001, -0.01:0.01:0.00001]

x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)
zs =  (z)
previous_zs = (z)
ns = torch.zeros_like(z)

# transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
previous_zs = previous_zs.to(device)
ns = ns.to(device)

for i in range(200):
    zs_ = zs*zs + previous_zs + z

    if (i > 0):
        previous_zs = zs

    not_diverged = torch.abs(zs_) < 4.0
    ns += not_diverged.type(torch.FloatTensor)
    zs = zs_

#plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,10))

def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.numpy()))
plt.tight_layout(pad=0)
plt.show()