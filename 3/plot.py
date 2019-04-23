import matplotlib 
import matplotlib.pyplot as plt
y = [39, 41, 47, 63]
x = [500, 600,700, 800]
n = [224, 272, 297, 334]

fig, ax = plt.subplots()
ax.plot(x, y)
plt.xlabel('No. of sensors')
plt.ylabel('Percentege Area Reduced')
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.savefig('svsc.png')
y = [96, 87,41, 25]
x = [30,45,60,75]
n = [219, 364, 454,459]

fig, ax = plt.subplots()
ax.plot(x, y)
plt.xlabel('Hole Size')
plt.ylabel('Percentege Area Reduced')
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.savefig('cvsc.png')
