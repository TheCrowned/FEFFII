import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.style.use('ggplot')

#Define plotting regin
fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(-3, 3), ylim=(-1, 1))

#Generate data
x = np.linspace(-3, 3, 91)
t = np.linspace(1, 25, 30)
X2, T2 = np.meshgrid(x, t)
 
sinT2 = np.sin(2*np.pi*T2/T2.max())
F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))

#Fraw first line
line = ax.plot(x, F[0, :], color='k', lw=2)[0]

#Update line y data
def animate(i):
    line.set_ydata(F[i, :])
   
#Execute and draw
anim = FuncAnimation(fig, animate, interval=100, frames=len(t)-1)

#anim.save('filename.mp4')
 
plt.draw()
plt.show()
