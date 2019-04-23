import matplotlib 
import matplotlib.pyplot as plt
y = [39, 53, 53, 52, 62,64]
x = [500, 600,700, 800, 900,1000]
n = [228, 314, 343, 394,354,343]

fig, ax = plt.subplots()
ax.plot(x, y)
plt.xlabel('No. of sensors')
plt.ylabel('Percentege Area Reduced')
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.savefig('svsc.png')
y = [98, 86,45, 20]
x = [30,45,60,75]
n = [196, 342, 352,362]

fig, ax = plt.subplots()
ax.plot(x, y)
plt.xlabel('Hole Size')
plt.ylabel('Percentege Area Reduced')
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.savefig('cvsc.png')
