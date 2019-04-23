import random
from math import sqrt,floor,atan2,sin,cos,degrees,radians
import numpy as np
import time
sthn=input().split(' ')#size times and holesize
canvas_size,times,holesize,sensor_count=int(sthn[0]),int(sthn[1]),int(sthn[2]),int(sthn[3])
canvas=int()
canvasX=canvas_size
canvasY=canvas_size
radius=10  
msensors=[]
intersections=[]
boundrySensors=[]
uncovered_boxes=set()

class Sensor :
    def __init__(self, position, radius):
        self.x=position[0]
        self.y=position[1]
        self.r=radius
        self.neighbours=[]
        self.isBoundry=False
        self.boundryPoints=[]
        self.boundryArc=[]
        self.maskedIntersections=[]
    def clear_sensor_data(self):
        self.neighbours.clear()
        self.isBoundry=False
        self.boundryPoints.clear()
        self.boundryArc.clear()
        self.maskedIntersections.clear()


def ScatterSensors(sensors):
    sensors.clear()
    uncovered_boxes.clear()
    for i in range(0,canvasX):
        for j in range (0,canvasY):
            uncovered_boxes.add((i,j))
    while len(uncovered_boxes) > 0 :
        pos=random.sample(uncovered_boxes,1)
        (x,y)=(pos[0][0],pos[0][1])
        cb=floor(radius/1.414)
        for i in range(-cb,cb,1):
            for j in range (-cb,cb,1):
                if (x+i,y+j) in uncovered_boxes:
                    uncovered_boxes.remove((x+i,y+j)) 

        sensors.append(Sensor(pos[0],radius))
    ct=len(sensors)
    if ct > sensor_count:
        print("insufficien sensors "+ct.__str__())
    while ct < sensor_count:
        pos=random.randint(0,canvasX),random.randint(0,canvasX)
        ct=ct+1
        sensors.append(Sensor(pos,radius))


def GetIntersections(sensors):
    intersections.clear()
    for sensor in sensors:
        sensor.neighbours.clear()
    for i in range(0,len(sensors)):
        for j in range(i+1,len(sensors)):
            x1,y1,r1 = sensors[i].x,sensors[i].y,sensors[i].r
            x2,y2,r2 = sensors[j].x,sensors[j].y,sensors[j].r
            dx,dy = x2-x1,y2-y1
            d = sqrt(dx*dx+dy*dy)
            if d > r1+r2:
                continue # no solutions, the circles are separate
            if d < abs(r1-r2):
                continue # no solutions because one circle is contained within the other
            if d == 0 and r1 == r2:
                continue # circles are coincident and there are an infinite number of solutions
            a = (r1*r1-r2*r2+d*d)/(2*d)
            h = sqrt(r1*r1-a*a)
            xm = x1 + a*dx/d
            ym = y1 + a*dy/d
            xs1 = xm + h*dy/d
            xs2 = xm - h*dy/d
            ys1 = ym - h*dx/d
            ys2 = ym + h*dx/d
            v1=(xs1 >= 0 and xs1 <canvasX) and (ys1 >= 0 and ys1 <canvasY)
            v2=(xs2 >= 0 and xs2 <canvasX) and (ys2 >= 0 and ys2 <canvasY)
            if d== r1+r2:
                intersections.append((xs1,ys1))
                sensors[i].neighbours.append([(j,xs1,ys1)])
                sensors[j].neighbours.append([(i,xs1,ys1)])
            else :
                if v1 and v2 :
                    intersections.append((xs1,ys1))
                    intersections.append((xs2,ys2))
                    sensors[i].neighbours.append([(j,xs1,ys1),(j,xs2,ys2)])
                    sensors[j].neighbours.append([(i,xs1,ys1),(i,xs2,ys2)])
                elif v1:
                    intersections.append((xs1,ys1))
                    sensors[i].neighbours.append([(j,xs1,ys1)])
                    sensors[j].neighbours.append([(i,xs1,ys1)])
                elif v2:
                    intersections.append((xs2,ys2))
                    sensors[i].neighbours.append([(j,xs2,ys2)])
                    sensors[j].neighbours.append([(i,xs2,ys2)])

def getHoleBoundary(sensors):
    GetIntersections(sensors)
    for sensor in sensors:
        sensor.isBoundry=False
        sensor.boundryPoints.clear()
        for i in range(0,len(sensor.neighbours)):
            for intrsxn in sensor.neighbours[i]:
                isInside=False
                for j in range(0,len(sensor.neighbours)):
                    if i != j:
                        #sensor.neighbours[j][0][0] is sensor number of neighbour bring checked
                        dx=abs(intrsxn[1]-sensors[sensor.neighbours[j][0][0]].x)
                        dy=abs(intrsxn[2]-sensors[sensor.neighbours[j][0][0]].y)
                        dist=sqrt(dx**2+dy**2)
                        if dist < sensors[sensor.neighbours[j][0][0]].r:
                            isInside=True
                            break
                if isInside == False:
                    sensor.isBoundry=True
                    sensor.boundryPoints.append(intrsxn)
                    
def getBoundryArc(sensors):
    for sensor in sensors:
        sensor.boundryArc.clear()   
    for sensor in sensors:
        if sensor.isBoundry == True:
            if len(sensor.boundryPoints) > 2:   #behaviour not defined when more than 
               pass
               # print("shit may happen")        #two boundary points exist
            elif len(sensor.boundryPoints) < 2:
                sensor.boundryPoints.clear()
                sensor.isBoundry=False
                continue
            p1=sensor.boundryPoints[0]
            p2=sensor.boundryPoints[1]
            dx1=p1[1]-sensor.x
            dy1=sensor.y-p1[2]#reverse y direction
            dx2=p2[1]-sensor.x
            dy2=sensor.y-p2[2]
            theta1=degrees(atan2(dy1,dx1))
            theta2=degrees(atan2(dy2,dx2))
            if(theta1 < 0):
                theta1=theta1+360
            if(theta2 < 0):
                theta2=theta2+360            
            if theta2 < theta1:
                theta1,theta2=theta2,theta1
            
            mtheta=(theta1+theta2)/2
            xm=sensor.x +sensor.r*cos(radians(mtheta))
            ym=sensor.y -sensor.r*sin(radians(mtheta))
            #area.create_circle(xm*SF,ym*SF,SF,fill='pink')
            isAnticlock=False
            for j in range(0,len(sensor.neighbours)):
                dx=abs(xm-sensors[sensor.neighbours[j][0][0]].x)
                dy=abs(ym-sensors[sensor.neighbours[j][0][0]].y)
                dist=sqrt(dx**2+dy**2)
                if dist < sensors[sensor.neighbours[j][0][0]].r:
                    isAnticlock=True
                    break
            if isAnticlock == False:
                sensor.boundryArc.append(theta1)
                sensor.boundryArc.append(theta2)
            else:
                sensor.boundryArc.append(theta2)
                sensor.boundryArc.append(theta1)
            
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def getHoleArea(sensors):
    s=sensors[0]
    j=0
    while s.isBoundry == False:
        j=j+1
        s=sensors[j]
    Pt=[]
    boundrySensors.clear()
    boundrySensors.append(j)
    p1=s.boundryPoints[0]
    Pt.append(p1)
    p=s.boundryPoints[1]
    sArea=0
    while p[1]!=p1[1] or p[2]!=p1[2]:
        Pt.append(p)
        sp=sensors[p[0]]
        boundrySensors.append(p[0])
        if sp.boundryPoints[0][1] == p[1] and sp.boundryPoints[0][2] == p[2]:
            p=sp.boundryPoints[1]
        else :
            p=sp.boundryPoints[0]
        #Angle of arc on boundary
        arcl=sp.boundryArc[1]-sp.boundryArc[0]
        if (arcl<0):
            arcl=arcl+360        
        #area of segment = area of sector - area of triangle for minor segment
        #area of segment = area of sector + area of triangle for major segment
        sArea=sArea + np.pi*sp.r*sp.r*arcl/360
        if arcl > 180:
            sArea=sArea + PolyArea([sp.x,sp.boundryPoints[0][1],sp.boundryPoints[1][1]],
                                [sp.y,sp.boundryPoints[0][2],sp.boundryPoints[1][2]])
        else:
            sArea=sArea - PolyArea([sp.x,sp.boundryPoints[0][1],sp.boundryPoints[1][1]],
                                [sp.y,sp.boundryPoints[0][2],sp.boundryPoints[1][2]])            
    X = [i[1] for i in Pt]
    Y = [i[2] for i in Pt]
    pArea=PolyArea(X,Y)
    hArea=pArea-sArea
    if hArea < 0:
        raise Exception("area calculation error")
    return hArea

def getApproxArea(sensors):
    partlyCovered=[]
    unCovered=[]
    for x in range(0,canvasX):
        for y in range(0,canvasY):
            isPartCovered=False
            isCovered=False
            for sensor in sensors:
                sx,sy,sr=sensor.x,sensor.y,sensor.r
                a= sqrt((sx-x)**2+(sy-y)**2) < sr
                b= sqrt((sx-(x+1))**2+(sy-y)**2) < sr
                c= sqrt((sx-x)**2+(sy-(y+1))**2) < sr
                d= sqrt((sx-(x+1))**2+(sy-(y+1))**2) < sr
                if (a and b and c and d):
                    isCovered=True
                    break
                elif(a or b or c or d):
                    isPartCovered=True
            if (isPartCovered==True and  isCovered==False):
                partlyCovered.append((x,y))
            if (isCovered==False and  isPartCovered==False):
                unCovered.append((x,y))
    area=len(unCovered)
    for a in partlyCovered:
        (x,y)=a
        for i in range (0,10):
            for j in range(0,10):
                px=x+i/10
                py=y+j/10
                isPartCovered=False
                isCovered=False
                for sensor in sensors:
                    sx,sy,sr=sensor.x,sensor.y,sensor.r
                    a= sqrt((sx-px)**2+(sy-py)**2) < sr
                    b= sqrt((sx-(px+1))**2+(sy-py)**2) < sr
                    c= sqrt((sx-px)**2+(sy-(py+1))**2) < sr
                    d= sqrt((sx-(px+1))**2+(sy-(py+1))**2) < sr
                    if (a and b and c and d):
                        isCovered=True
                        break
                    elif(a or b or c or d):
                        isPartCovered=True
                if (isPartCovered==True and  isCovered==False):
                    area=area+0.004
                if (isCovered==False and  isPartCovered==False):
                    area=area+0.01
    return area

def getMaskedIntersections(sensors):
    for i in range(0,len(sensors)):
        sensor=sensors[i]
        sensor.maskedIntersections.clear()
        if sensor.isBoundry==True:
            #All masked intrsxns will be intersection of its neighbours
            for nb in sensor.neighbours:
                nbIndex=nb[0][0]
                ns=sensors[nbIndex]
                for k in range (0,len(ns.neighbours)):
                    for intrsxn in ns.neighbours[k]:
                        isInside=False
                        dx = sensor.x - intrsxn[1]
                        dy = sensor.y - intrsxn[2]
                        dist = sqrt(dx**2 + dy**2)
                        #intersection is not masked by if its not covered by s
                        if dist > sensor.r:
                            continue
                        for j in range(0,len(ns.neighbours)):
                            if ns.neighbours[j][0][0]!=i and j!=k:
                                dx=abs(intrsxn[1]-sensors[ns.neighbours[j][0][0]].x)
                                dy=abs(intrsxn[2]-sensors[ns.neighbours[j][0][0]].y)
                                dist=sqrt(dx**2+dy**2)
                                if dist < sensors[ns.neighbours[j][0][0]].r:
                                    if sensors[ns.neighbours[j][0][0]].isBoundry==False:
                                        isInside=True
                                        break 
                        if isInside == False:
                            if (intrsxn[1],intrsxn[2]) not in sensor.maskedIntersections:
                                sensor.maskedIntersections.append((intrsxn[1],intrsxn[2]))                               
#where line of slope m is passing through y1 x1
#and circle of radius r at x2,y2
def getLineCircleIntersection(m,y1,x1,y2,x2,r,isVertical):
    if isVertical:
        yi1=y2-sqrt(r**2-(x1-x2)**2)
        yi2=y2-sqrt(r**2-(x1-x2)**2)
        return [(x1,yi1),(x1,yi2)]
    else:    
        c=y1-m*x1
        aq=1+m**2
        bq=-2*x2 + 2*m*(c-y2)
        cq=x2**2 +(c-y2)**2 -r**2
        xi1=(-bq+sqrt(bq**2-4*aq*cq))/(2*aq)
        xi2=(-bq-sqrt(bq**2-4*aq*cq))/(2*aq)
        yi1=m*xi1+c
        yi2=m*xi2+c 
        return [(xi1,yi1),(xi2,yi2)]

def getnewcentre(sc,spr,sn,sensors):
    scurr=sensors[boundrySensors[sc]]
    sprev=sensors[boundrySensors[spr]]
    if sn ==len(boundrySensors):
        sn=0
    snext=sensors[boundrySensors[sn]]
    dx=snext.x-sprev.x
    dy=snext.y-sprev.y
    TC=[]   # new tentative centres
    for p in scurr.maskedIntersections:
        bp= [(i[1],i[2]) for i in scurr.boundryPoints]
        if p in bp:
            continue   
        try:
            m=-dx/dy
        except :
            pts=getLineCircleIntersection(0,scurr.y,scurr.x,p[1],p[0],scurr.r,True)
        else:
            pts=getLineCircleIntersection(m,scurr.y,scurr.x,p[1],p[0],scurr.r,False)
        for pt in pts:
            if abs(pt[0]-scurr.x)<0.000004 and abs(pt[1]-scurr.y)<0.000004:
                continue
            agl=degrees(atan2(scurr.y-pt[1],pt[0]-scurr.x))
            if agl < 0:
                agl=agl+360
            
            if abs(agl - (scurr.boundryArc[1] + scurr.boundryArc[0])/2) < 90:
                TC.append(pt)        
    mindist=10000000
    minpos=(sensors[boundrySensors[0]].x,sensors[boundrySensors[0]].y)
    for p in TC:
        dist=sqrt((scurr.x-p[0])**2+(scurr.y-p[1])**2)
        if dist < mindist:
            mindist=dist
            minpos=p
    return (minpos)

def shrinkHole(sensors):
    l=len(boundrySensors)
    dist=0
    caldist= lambda a,b,c,d:sqrt((a-c)**2+(a-c)**2)
    if l%2 == 1:
        for i in range(1,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors))
            dist=dist+caldist(sensors[boundrySensors[i]].x,sensors[boundrySensors[i]].y,tmp[0],tmp[1])
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1]           
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)
        for i in range(2,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors))
            dist=dist+caldist(sensors[boundrySensors[i]].x,sensors[boundrySensors[i]].y,tmp[0],tmp[1])
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1] 
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)
        c=getnewcentre(0,-1,1,sensors)   
        dist=dist+caldist(sensors[boundrySensors[i]].x,sensors[boundrySensors[i]].y,tmp[0],tmp[1])
        sensors[boundrySensors[0]].x=c[0]
        sensors[boundrySensors[0]].y=c[1]        
    else:
        for i in range(0,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors))
            dist=dist+caldist(sensors[boundrySensors[i]].x,sensors[boundrySensors[i]].y,tmp[0],tmp[1])
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1] 
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)
        for i in range(1,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors))
            dist=dist+caldist(sensors[boundrySensors[i]].x,sensors[boundrySensors[i]].y,tmp[0],tmp[1])            
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1] 
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)
    return dist
def shrinkOne(sensors):
        c=getnewcentre(0,-1,1,sensors)       
        sensors[boundrySensors[0]].x=c[0]
        sensors[boundrySensors[0]].y=c[1]


i=0
pr=0
tcc=0 
tdt=0
print("For Holesize "+holesize.__str__())
print("Number of sensor "+sensor_count.__str__())
while i < times:
    try:    
        msensors.clear()
        ScatterSensors(msensors)
        xp=canvasX/2
        yp=canvasY/2
        tsensors=[]
        for sensor in msensors:
            dx=abs(sensor.x-xp)
            dy=abs(sensor.y-yp)
            if dx >= holesize/2 or dy >= holesize/2 :
                tsensors.append(sensor)
        msensors.clear()
        msensors=tsensors
        getHoleBoundary(msensors)
        getBoundryArc(msensors)
        ia=getHoleArea(msensors)
        getMaskedIntersections(msensors)
        dt=shrinkHole(msensors)
        tdt=dt+tdt
        for sensor in msensors:
            sensor.clear_sensor_data()
        getHoleBoundary(msensors)
        getBoundryArc(msensors)
        fa=getApproxArea(msensors)
        if fa > ia:
            raise "some error must have occured"
        pr=pr+(ia-fa)/ia*100
        if (fa==0):
            tcc=tcc+1
        print(ia.__str__()+" "+fa.__str__()+" "+dt.__str__())
        i=i+1 
    except:
        continue

print("number of times hole fully covered "+tcc.__str__())
print("average area reduced = "+(pr/times).__str__())
print("average distance traveled = "+(tdt/times).__str__())