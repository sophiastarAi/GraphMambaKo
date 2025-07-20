import matplotlib.pyplot as plt
import numpy as np


time = np.linspace(0, 10, 100)
amplitude = np.sin(time)


plt.figure(figsize=(10, 5))


plt.plot(time, amplitude, label='Dynamic Mode')


plt.title('Dynamic Mode Decomposition Visualization')
plt.xlabel('Time')
plt.ylabel('Amplitude')


plt.legend()


plt.show()