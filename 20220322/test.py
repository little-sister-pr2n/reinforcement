import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi)
sn = np.sin(x)
cs = np.cos(x)

fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))

axs[0].plot(sn, c='b', label="sin x")
handles0, labels0 = axs[0].get_legend_handles_labels()

axs[1].plot(cs, c='r', label="cos x")
handles1, labels1 = axs[1].get_legend_handles_labels()

fig.legend(handles0 + handles1, labels0 + labels1, ncol=2, loc='upper center')

plt.show()