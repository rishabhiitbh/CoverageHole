import os
import re
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
y = []
x = []
n = []
model=LinearRegression()
files = [f for f in os.listdir('.') if os.path.isfile(f)]
SvsCdata =[name for name in files if name[:4]=='SvsC' ]
CvsCdata =[name for name in files if name[:4]=='CvsC' ]
SvsCdata=sorted(SvsCdata, key=lambda x:int((x[:-4])[4:]))
CvsCdata=sorted(CvsCdata, key=lambda x:int((x[:-4])[4:]))

for f in SvsCdata:
    x.append(int((f[:-4])[4:]))
    with open(f) as file:
        for line in file:
            if 'area' in line:
                words=line.split(' ')
                y.append(int(float(words[4])))
            if 'traveled' in line:
                words=line.split(' ')
                n.append(int(float(words[4])))

xn=np.linspace(x[0], x[-1], num=200, endpoint=True)
#ip=interp1d(x,y,kind='cubic')
reg=model.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))
fig, ax = plt.subplots()

ax.plot(x,y,xn,model.predict(np.array(xn).reshape(-1,1)).reshape(-1))
plt.xlabel('No. of sensors')
plt.ylabel('Percentege Area Reduced')
plt.legend(['original data','Linear regression'])
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.savefig('svsc_linreg.png')
x.clear()
y.clear()
n.clear()

for f in CvsCdata:
    x.append(int((f[:-4])[4:]))
    with open(f) as file:
        for line in file:
            if 'area' in line:
                words=line.split(' ')
                y.append(int(float(words[4])))
            if 'traveled' in line:
                words=line.split(' ')
                n.append(int(float(words[4])))

xn=np.linspace(x[0], x[-1], num=200, endpoint=True)
#ip=interp1d(x,y,kind='cubic')
reg=model.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))
fig, ax = plt.subplots()
ax.plot(x,y,'o',xn,model.predict(np.array(xn).reshape(-1,1)).reshape(-1))
plt.legend(['original data','Linear regression'])
plt.xlabel('Hole Size')
plt.ylabel('Percentege Area Reduced')
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.savefig('cvsc_linreg.png')
