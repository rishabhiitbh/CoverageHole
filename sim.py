import tkinter as tk
import random
from math import sqrt,floor,atan2,sin,cos,degrees,radians
import numpy as np
import time
import pudb

canvasX=100
canvasY=100
SF=10  # Scale factor for displaying
radius=6  
msensors=[]
intersections=[]
Holes=[]
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

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle

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

def GetIntersections(sensors):
    intersections.clear()
    for sensor in sensors:
        sensor.neighbours.clear()
    for i in range(0,len(sensors)):
        for j in range(i+1,len(sensors)):
            x1,y1,r1 = sensors[i].x,sensors[i].y,sensors[i].r
            x2,y2,r2 = sensors[j].x,sensors[j].y,sensors[j].r
            dx,dy = x2-x1,y2-y1
            dsq = dx*dx+dy*dy
            d=sqrt(dsq)
            if dsq > (r1+r2)**2:
                continue # no solutions, the circles are separate
            if dsq < (r1-r2)**2:
                continue # no solutions because one circle is contained within the other
            if dsq == 0 and r1 == r2:
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

def insideTriangle(sensors,triSens,point):
    s=[sensors[x] for x in triSens]
    if ((s[0].x==s[1].x and s[0].y==s[1].y) or 
        (s[1].x==s[2].x and s[1].y==s[2].y) or 
        (s[0].x==s[2].x and s[0].y==s[2].y)):
        return False #duplicate sensors
    p=[]
    for k in range(0,len(triSens)):
        i=(k+1)%len(triSens)
        for n in s[k].neighbours:
            if n[0][0]==triSens[i]:
                if len(n)==1:
                    return False #only one point of intersection between them
                else:
                    if abs(n[0][1]-point[1]) < 0.0000000001 and abs(n[0][2]-point[2]) < 0.0000000001:
                        p.append((n[1][1],n[1][2]))
                    else:
                        p.append((n[0][1],n[0][2]))
    #s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py);
    #t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py);
    #Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y);
    #Just evaluate s, t and 1-s-t. The point p is inside the triangle if and only if they are all positive.
    Area = 0.5 *(-p[1][1]*p[2][0] + p[0][1]*(-p[1][0] + p[2][0]) + p[0][0]*(p[1][1] - p[2][1]) + p[1][0]*p[2][1])
    if Area==0:
        return False
    t = 1/(2*Area)*(p[0][0]*p[1][1] - p[0][1]*p[1][0] + (p[0][1] - p[1][1])*point[1] + (p[1][0] - p[0][0])*point[2])
    s = 1/(2*Area)*(p[0][1]*p[2][0] - p[0][0]*p[2][1] + (p[2][1] - p[0][1])*point[1] + (p[0][0] - p[2][0])*point[2])
    if s >=0 and t >=0 and 1-s-t >= 0:
        return True
    return False

def getHoleBoundary(sensors):
    GetIntersections(sensors)
    for k in range(0,len(sensors)):
        sensor=sensors[k]
        sensor.isBoundry=False
        sensor.boundryPoints.clear()
        for i in range(0,len(sensor.neighbours)):
            for a in range(0,len(sensor.neighbours[i])):
                intrsxn=sensor.neighbours[i][a]
                isInside=False
                for j in range(0,len(sensor.neighbours)):
                    if i != j:
                        #sensor.neighbours[j][0][0] is sensor number of neighbour bring checked
                        dx=abs(intrsxn[1]-sensors[sensor.neighbours[j][0][0]].x)
                        dy=abs(intrsxn[2]-sensors[sensor.neighbours[j][0][0]].y)
                        distsq=dx**2+dy**2
                        if abs(distsq-sensors[sensor.neighbours[j][0][0]].r**2)<0.0000000001:
                            #probably a triple intersection
                            isInside=insideTriangle(sensors,[k,sensor.neighbours[j][0][0],
                                                            sensor.neighbours[i][0][0]],intrsxn)
                            if isInside==True:
                                sensor.neighbours[i][a]=sensor.neighbours[i][a]+(sensor.neighbours[j][0][0],)
                                break
                        elif distsq < (sensors[sensor.neighbours[j][0][0]].r**2) + 0.0000000001 :#because float thats why
                            isInside=True
                            sensor.neighbours[i][a]=sensor.neighbours[i][a]+(sensor.neighbours[j][0][0],)
                            break
                if isInside == False:
                    sensor.isBoundry=True
                    sensor.boundryPoints.append(intrsxn)
                    
def getBoundryArc(sensors):
    for sensor in sensors:
        sensor.boundryArc.clear()   
    for i in range(0,len(sensors)):
        angles=[]
        sensor=sensors[i]
        if sensor.isBoundry == True:
            if len(sensor.boundryPoints) < 2:
                # sensor.boundryPoints.clear()
                debugPlot(sensors,i)
                pudb.set_trace()
                sensor.isBoundry=False
                continue
            for point in sensor.boundryPoints:
                dx=point[1]-sensor.x
                dy=sensor.y-point[2]#reverse y direction
                theta=degrees(atan2(dy,dx))
                if theta < 0:
                    theta=theta+360
                angles.append((theta,point))#neighbour point
            angles=sorted(angles,key=lambda x: x[0])
            num=len(angles)
            medians=[]
            for k in range(0,num):
                theta1=angles[k][0]
                theta2=angles[(k+1)%num][0]
                if theta1 < theta2:
                    mtheta=(theta1+theta2)/2
                elif theta1 > theta2:
                    mtheta=(theta1+theta2)/2 - 180
                    if mtheta < 0:
                        mtheta=mtheta+360
                else:
                    continue
                xm=sensor.x +sensor.r*cos(radians(mtheta))
                ym=sensor.y -sensor.r*sin(radians(mtheta))
                medians.append((xm,ym))
                #area.create_circle(xm*SF,ym*SF,SF,fill='pink')
                isBoundaryArc=True
                for j in range(0,len(sensor.neighbours)):
                    dx=abs(xm-sensors[sensor.neighbours[j][0][0]].x)
                    dy=abs(ym-sensors[sensor.neighbours[j][0][0]].y)
                    dist=sqrt(dx**2+dy**2)
                    coinc=[]
                    circ=[]
                    if abs(dist-sensors[sensor.neighbours[j][0][0]].r) < 0.0000000001:
                        ts=sensors[sensor.neighbours[j][0][0]]
                        if (ts.x,ts.y) not in coinc:
                            coinc.append((ts.x,ts.y))
                            circ.append(sensor.neighbours[j][0][0])
                    elif dist < sensors[sensor.neighbours[j][0][0]].r:
                        isBoundaryArc=False
                        break
                
                if isBoundaryArc and len(circ)>=2:
                    isInside=insideTriangle(sensors,[i,circ[0],circ[1]],(0,xm,ym))
                    if isInside==True:
                        isBoundaryArc=False
                if isBoundaryArc:
                    sensor.boundryArc.append((angles[k],angles[(k+1)%num]))
            if len(sensor.boundryArc)==0:
                debugPlot(sensors,i,medians)
                pudb.set_trace()
                

def debugPlot(sensors,sensorNum,extraPoints=[]):
    area.delete('all')
    area.update()
    area.update_idletasks()
    for sensor in sensors:
        area.create_circle(sensor.x*SF,sensor.y*SF,sensor.r*SF,outline="black",width=1)

    for x in sensors[sensorNum].neighbours:
        sensor=sensors[x[0][0]]
        area.create_circle(sensor.x*SF,sensor.y*SF,sensor.r*SF,outline="red",width=1)
        area.create_text(sensor.x*SF,sensor.y*SF,text=x[0][0].__str__(),fill="pink")
        for intersection in x:
            area.create_circle(intersection[1]*SF,intersection[2]*SF,SF/20,fill='red')
            if len(intersection)==4:
                area.create_text(intersection[1]*SF,intersection[2]*SF,text=intersection[3].__str__())
    sensor=sensors[sensorNum]
    area.create_circle(sensor.x*SF,sensor.y*SF,sensor.r*SF,outline="green",width=1)
    for sb in sensor.boundryPoints:
        area.create_circle(sb[1]*SF,sb[2]*SF,SF/2,outline="blue",width=1)
    for ep in extraPoints:
        area.create_circle(ep[0]*SF,ep[1]*SF,SF/5,fill='yellow')
    area.update()
    area.update_idletasks()


    
def drawPlot(sensors):
    for sensor in sensors:
        if sensor.isBoundry == False:
            area.create_circle(sensor.x*SF,sensor.y*SF,sensor.r*SF,outline="#000",width=1) 
        else:
            area.create_circle(sensor.x*SF,sensor.y*SF,sensor.r*SF,outline="#000",width=1)
            for arcs in sensor.boundryArc:
                arcl=arcs[1][0]-arcs[0][0]
                if (arcl<0):
                    arcl=arcl+360
                area.create_arc((sensor.x-sensor.r)*SF,
                                (sensor.y-sensor.r)*SF,
                                (sensor.x+sensor.r)*SF,
                                (sensor.y+sensor.r)*SF,
                                start=arcs[0][0],
                                extent=arcl,
                                outline="blue",style=tk.ARC,
                                width=2)
            for intrsxn in sensor.maskedIntersections:
                area.create_circle(intrsxn[0]*SF,intrsxn[1]*SF,SF/2,outline="blue",width=1)  
    for intersection in intersections:
        area.create_circle(intersection[0]*SF,intersection[1]*SF,SF/5,fill='red')
        
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def getHoles(sensors):
    Holes.clear()
    visitedarcs=[]
    for i  in range (0,len(sensors)):
        if sensors[i].isBoundry is True:
            HoleArcs=[]
            for arcs in sensors[i].boundryArc:
                if arcs not in visitedarcs:
                    visitedarcs.append(arcs)
                    initArc=arcs
                    p1=initArc[0][1]
                    p=initArc[1][1]
                    sensorNum=i
                    currArc=initArc
                    HoleArcs.append((currArc,sensorNum,(p[1],p[2])))
                    while p[1]!=p1[1] or p[2]!=p1[2]: 
                        if (currArc,sensorNum,(p[1],p[2])) not in HoleArcs:
                            HoleArcs.append((currArc,sensorNum,(p[1],p[2])))
                        sensorNum=p[0]     
                        for arc in sensors[sensorNum].boundryArc:
                            if abs(arc[0][1][1]-p[1])<0.00000000001 and abs(arc[0][1][2]-p[2])<0.00000000001:
                                p=arc[1][1]
                                currArc=arc
                                visitedarcs.append(currArc)
                                break
                            elif abs(arc[1][1][1]-p[1])<0.00000000001 and abs(arc[1][1][2]-p[2])<0.00000000001:
                                p=arc[0][1]
                                currArc=arc
                                visitedarcs.append(currArc)
                                break
                        if sensorNum is p[0]: #if no next sensor was found.... this should not occur
                            pudb.set_trace()        
                    if (currArc,sensorNum,(p[1],p[2])) not in HoleArcs:
                        HoleArcs.append((currArc,sensorNum,(p[1],p[2])))
                if len(HoleArcs)!=0:                
                    Holes.append(HoleArcs.copy(),)
                    HoleArcs.clear()

def getHoleArea(sensors,Holes):
    totalArea=0
    for hole in Holes:
        sArea=0
        for arc in hole:
            arcl=arc[0][1][0]-arc[0][0][0]
            if (arcl<0):
                arcl=arcl+360
            sp=sensors[arc[1]]       
            #area of segment = area of sector - area of triangle for minor segment
            #area of segment = area of sector + area of triangle for major segment
            sArea=sArea + np.pi*sp.r*sp.r*arcl/360
            if arcl > 180:
                sArea=sArea + PolyArea([sp.x,arc[0][0][1][1],arc[0][1][1][1]],
                                    [sp.y,arc[0][0][1][2],arc[0][1][1][2]])
            else:
                sArea=sArea -  PolyArea([sp.x,arc[0][0][1][1],arc[0][1][1][1]],
                                    [sp.y,arc[0][0][1][2],arc[0][1][1][2]])            
        X = [i[2][0] for i in hole]
        Y = [i[2][1] for i in hole]
        pArea=PolyArea(X,Y)
        hArea=pArea-sArea
        if hArea < -0.01:
            raise Exception("area calculation error")
        totalArea=totalArea+hArea
    print("area of hole is "+totalArea.__str__())


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
    print("the approximated area is "+area.__str__())

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
        try:
            xi1=(-bq+sqrt(bq**2-4*aq*cq))/(2*aq)
            xi2=(-bq-sqrt(bq**2-4*aq*cq))/(2*aq)
        except ValueError:
            return None
        yi1=m*xi1+c
        yi2=m*xi2+c 
        return [(xi1,yi1),(xi2,yi2)]
#type 0 for our 
def getnewcentre(sc,spr,sn,sensors,boundrySensors,stratType):
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
        if stratType == 0:    
            try:
                m=-dx/dy
            except :
                pts=getLineCircleIntersection(0,scurr.y,scurr.x,p[1],p[0],scurr.r,True)
            else:
                pts=getLineCircleIntersection(m,scurr.y,scurr.x,p[1],p[0],scurr.r,False)

        elif stratType == 1:
            arc=scurr.boundryArc[0]
            arcl=arc[1][0]-arc[0][0]
            if (arcl<0):
                arcl=arcl+360
            m=(arc[0][0]+(random.random()*arcl))%360
            m=360-m # take mirror image of arc because tkinter
            pts=getLineCircleIntersection(m,scurr.y,scurr.x,p[1],p[0],scurr.r,m==90 or m==270)
        else:
            pts=None
        
        
        if pts==None:
            return (scurr.x,scurr.y)
        for pt in pts:
            if abs(pt[0]-scurr.x)< 0 and abs(pt[1]-scurr.y) < 0:
                continue
            agl=degrees(atan2(scurr.y-pt[1],pt[0]-scurr.x))
            if agl < 0:
                agl=agl+360
            
            if abs(agl - (scurr.boundryArc[0][1][0] + scurr.boundryArc[0][0][0])/2) < 90:
                TC.append(pt)        
    mindist=10000000
    minpos=(sensors[boundrySensors[0]].x,sensors[boundrySensors[0]].y)
    for p in TC:
        dist=sqrt((scurr.x-p[0])**2+(scurr.y-p[1])**2)
        if dist < mindist:
            mindist=dist
            minpos=p
    return (minpos)

def shrinkHole(sensors,Hole,stratType):
    boundrySensors=[arc[1] for arc in Hole]
    l=len(boundrySensors)
    if l%2 == 1:
        for i in range(1,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors,boundrySensors,stratType))# also send the arc for random case
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1] 
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)
        for i in range(2,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors,boundrySensors,stratType))
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1] 
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)
        c=getnewcentre(0,-1,1,sensors,boundrySensors,stratType)       
        sensors[boundrySensors[0]].x=c[0]
        sensors[boundrySensors[0]].y=c[1]        
    else:
        for i in range(0,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors,boundrySensors,stratType))
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1] 
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)
        for i in range(1,len(boundrySensors),2):
            tmp=(getnewcentre(i,i-1,(i+1)%l,sensors,boundrySensors,stratType))
            sensors[boundrySensors[i]].x=tmp[0]
            sensors[boundrySensors[i]].y=tmp[1] 
            getHoleBoundary(sensors)
            getBoundryArc(sensors)
            getMaskedIntersections(sensors)


def repat(sensors):
    area.delete('all')
    area.update()
    area.update_idletasks()
    shrinkHole(sensors,Holes[0],stratType=1)
    for sensor in sensors:
        sensor.clear_sensor_data()
    getHoleBoundary(sensors)
    getBoundryArc(sensors)
    getHoles(sensors)
    getMaskedIntersections(sensors)
    drawPlot(sensors)
    getHoleArea(sensors,Holes)
    #getApproxArea(sensors)

ScatterSensors(msensors)
        
gridx=canvasX*SF
gridy=canvasY*SF

window=tk.Tk()
window.title("SenSim")
area=tk.Canvas(window,height=gridy,width=gridx)
area.configure(background='white')
area.pack()
xp=30
yp=30

tsensors=[]
for sensor in msensors:
    dx=abs(sensor.x-xp)
    dy=abs(sensor.y-yp)
    dist=sqrt(dx**2+dy**2)
    if dist >= sensor.r+4:
        tsensors.append(sensor)
msensors.clear()
msensors=tsensors
getHoleBoundary(msensors)
getBoundryArc(msensors)
getHoles(msensors)
getHoleArea(msensors,Holes)
getMaskedIntersections(msensors)
drawPlot(msensors)
#getApproxArea(msensors)
area.after(2000,lambda : repat(msensors))  
window.mainloop()
