import tkinter as tk
import random
from math import sqrt, floor, atan2, sin, cos, degrees, radians
import numpy as np
import time
import pudb
from PIL import Image
import io
import os
import subprocess
ENABLE_PROXIMITY_EQUALITY = False

canvasX = 100
canvasY = 100
SF = 10  # Scale factor for displaying
radius = 10
msensors = []
intersections = []
Holes = []
uncovered_boxes = set()


class Point:
    def __init__(self, X, Y, data=None):
        self.x = X
        self.y = Y
        self.data = data

    def __eq__(self, other):
        if ENABLE_PROXIMITY_EQUALITY:
            return abs(self.x-other.x) < 0.0000001 and abs(self.y-other.y) < 0.0000001
        else:
            return self.x == other.x and self.y == other.y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise Exception("Point has only two dimension")

    def __iter__(self):
        yield self.x
        yield self.y


class Line:
    """represent a line as point and its slope"""

    def __init__(self, pivotPoint, slope=None):
        self.pivotPoint = pivotPoint
        self.isVertical = False if slope != None else True
        self.slope = slope


class LineSegment:
    def __init__(self, p1, p2):
        self.start = p1
        self.end = p2

    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end) or (self.end == other.start and self.start == other.end)

    def __getitem__(self, key):
        if key == 0:
            return self.start
        elif key == 1:
            return self.end
        else:
            raise Exception("Line Has only two points")

    def __iter__(self):
        yield self.start
        yield self.end

    def getLineObject(self):
        if self.start.x == self.end.x:
            return Line(self.start)
        else:
            return Line(self.start, (self.start.y-self.end.y)/(self.start.x-self.end.x))


class RegularHexagon:
    def __init__(self, centre, radius):
        d = (sqrt(3)/2)*radius
        self.radius = radius
        r = radius
        self.centre = Point(*centre)
        x = self.centre.x
        y = self.centre.y
        self.points = [Point(x, y+r), Point(x+d, y+r/2), Point(x+d, y-r/2),
                       Point(x, y-r), Point(x-d, y-r/2), Point(x-d, y+r/2)]

    def __getitem__(self, key):
        return self.points[key]

    def getPoints(self):
        return self.points.copy()

    def getLineSegments(self):
        lines = []
        for i in range(6):
            lines.append(LineSegment(self.points[i], self.points[i-1]))
        return lines


class NeighbourPoint:
    def __init__(self, nid, points):
        self.nid = nid
        self.points = points


class Sensor:
    def __init__(self, position, radius):
        self.x = position[0]
        self.y = position[1]
        self.r = radius
        self.neighbours = []
        self.boundaryLevel = -1  # still not calculated
        self.boundaryPoints = []
        self.boundaryArc = []
        self.maskedIntersections = []

    def clear_sensor_data(self):
        self.neighbours.clear()
        self.boundaryLevel = -1
        self.boundaryPoints.clear()
        self.boundaryArc.clear()
        self.maskedIntersections.clear()

    def getCentre(self):
        return Point(self.x, self.y)


def getLineCircleIntersection(line, centre, r):
    x1 = line.pivotPoint.x
    y1 = line.pivotPoint.y
    m = line.slope
    x2 = centre.x
    y2 = centre.y
    if line.isVertical:
        try:
            yi1 = y2-sqrt(r**2-(x1-x2)**2)
            yi2 = y2-sqrt(r**2-(x1-x2)**2)
            return [Point(x1, yi1), Point(x1, yi2)]
        except ValueError:
            return None
    else:
        c = y1-m*x1
        aq = 1+m**2
        bq = -2*x2 + 2*m*(c-y2)
        cq = x2**2 + (c-y2)**2 - r**2
        try:
            xi1 = (-bq+sqrt(bq**2-4*aq*cq))/(2*aq)
            xi2 = (-bq-sqrt(bq**2-4*aq*cq))/(2*aq)
        except ValueError:
            return None
        yi1 = m*xi1+c
        yi2 = m*xi2+c
        return [Point(xi1, yi1), Point(xi2, yi2)]


def getLineSegmentSensorIntersection(lineSegment, sensor):
    line = lineSegment.getLineObject()
    intersectionPoints = getLineCircleIntersection(
        line, sensor.getCentre(), sensor.r)
    result = []
    s = lineSegment.start
    e = lineSegment.end
    if intersectionPoints == None:
        return []
    for point in intersectionPoints:
        if min(s.x, e.x)-0.000001 <= point.x <= max(s.x, e.x)+0.0000001 and min(s.y, e.y)-0.000001 <= point.y <= max(s.y, e.y)+0.0000001:
            result.append(point)
    return result


def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)


tk.Canvas.create_circle = _create_circle


def ScatterSensors(sensors):
    sensors.clear()
    uncovered_boxes.clear()
    for i in range(0, canvasX):
        for j in range(0, canvasY):
            uncovered_boxes.add((i, j))
    while len(uncovered_boxes) > 0:
        pos = random.sample(uncovered_boxes, 1)
        (x, y) = (pos[0][0], pos[0][1])
        cb = floor(radius/1.414)
        for i in range(-cb, cb, 1):
            for j in range(-cb, cb, 1):
                if (x+i, y+j) in uncovered_boxes:
                    uncovered_boxes.remove((x+i, y+j))
        sensors.append(Sensor(pos[0], radius))


def GetIntersections(sensors):
    intersections.clear()
    for sensor in sensors:
        sensor.neighbours.clear()
    for i in range(0, len(sensors)):
        if sensors[i].boundaryLevel != -1:
            continue
        for j in range(i+1, len(sensors)):
            if sensors[i].boundaryLevel != -1:
                continue
            x1, y1, r1 = sensors[i].x, sensors[i].y, sensors[i].r
            x2, y2, r2 = sensors[j].x, sensors[j].y, sensors[j].r
            dx, dy = x2-x1, y2-y1
            dsq = dx*dx+dy*dy
            d = sqrt(dsq)
            if dsq > (r1+r2)**2:
                continue  # no solutions, the circles are separate
            if dsq < (r1-r2)**2:
                continue  # no solutions because one circle is contained within the other
            if dsq == 0 and r1 == r2:
                continue  # circles are coincident and there are an infinite number of solutions
            a = (r1*r1-r2*r2+d*d)/(2*d)
            h = sqrt(r1*r1-a*a)
            xm = x1 + a*dx/d
            ym = y1 + a*dy/d
            xs1 = xm + h*dy/d
            xs2 = xm - h*dy/d
            ys1 = ym - h*dx/d
            ys2 = ym + h*dx/d
            v1 = (xs1 >= 0 and xs1 < canvasX) and (ys1 >= 0 and ys1 < canvasY)
            v2 = (xs2 >= 0 and xs2 < canvasX) and (ys2 >= 0 and ys2 < canvasY)
            if d == r1+r2:
                intersections.append(Point(xs1, ys1))
                sensors[i].neighbours.append(
                    NeighbourPoint(j, [Point(xs1, ys1, j)]))
                sensors[j].neighbours.append(
                    NeighbourPoint(i, [Point(xs1, ys1, i)]))
            else:
                if v1 and v2:
                    intersections.append(Point(xs1, ys1))
                    intersections.append(Point(xs2, ys2))
                    sensors[i].neighbours.append(
                        NeighbourPoint(j, [Point(xs1, ys1, j), Point(xs2, ys2, j)]))
                    sensors[j].neighbours.append(
                        NeighbourPoint(i, [Point(xs1, ys1, i), Point(xs2, ys2, i)]))
                elif v1:
                    intersections.append(Point(xs1, ys1))
                    sensors[i].neighbours.append(
                        NeighbourPoint(j, [Point(xs1, ys1, j)]))
                    sensors[j].neighbours.append(
                        NeighbourPoint(i, [Point(xs1, ys1, i)]))
                elif v2:
                    intersections.append(Point(xs2, ys2))
                    sensors[i].neighbours.append(
                        NeighbourPoint(j, [Point(xs2, ys2, j)]))
                    sensors[j].neighbours.append(
                        NeighbourPoint(i, [Point(xs2, ys2, i)]))


def insideTriangle(sensors, triSens, point):
    s = [sensors[x] for x in triSens]
    if ((s[0].x == s[1].x and s[0].y == s[1].y) or
        (s[1].x == s[2].x and s[1].y == s[2].y) or
            (s[0].x == s[2].x and s[0].y == s[2].y)):
        return False  # duplicate sensors
    p = []
    for k in range(0, len(triSens)):
        i = (k+1) % len(triSens)
        for n in s[k].neighbours:
            if n.nid == triSens[i]:
                if len(n.points) == 1:
                    return False  # only one point of intersection between them
                else:
                    if n.points[0] == point:
                        p.append(n.points[1])
                    else:
                        p.append(n.points[0])
    # s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py);
    # t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py);
    # Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y);
    # Just evaluate s, t and 1-s-t. The point p is inside the triangle if and only if they are all positive.
    Area = 0.5 * (-p[1].y*p[2].x + p[0].y*(-p[1].x + p[2].x)
                                 + p[0].x*(p[1].y - p[2].y) + p[1].x*p[2].y)
    if Area == 0:
        return False
    t = 1/(2*Area)*(p[0].x*p[1].y - p[0].y*p[1].x +
                    (p[0].y - p[1].y)*point.x + (p[1].x - p[0].x)*point.y)
    s = 1/(2*Area)*(p[0].y*p[2].x - p[0].x*p[2].y +
                    (p[2].y - p[0].y)*point.x + (p[0].x - p[2].x)*point.y)
    if s >= 0 and t >= 0 and 1-s-t >= 0:
        return True
    return False


def getHoleBoundary(sensors):
    GetIntersections(sensors)
    for k in range(0, len(sensors)):
        sensor = sensors[k]
        sensor.boundaryLevel = -1
        sensor.boundaryPoints.clear()
        for i in range(0, len(sensor.neighbours)):
            for a in range(0, len(sensor.neighbours[i].points)):
                intrsxn = sensor.neighbours[i].points[a]
                isInside = False
                for j in range(0, len(sensor.neighbours)):
                    if i != j:
                        # sensor.neighbours[j] is sensor number of neighbour bring checked
                        dx = abs(
                            intrsxn.x-sensors[sensor.neighbours[j].nid].x)
                        dy = abs(
                            intrsxn.y-sensors[sensor.neighbours[j].nid].y)
                        distsq = dx**2+dy**2
                        if abs(distsq-sensors[sensor.neighbours[j].nid].r**2) < 0.0000000001:
                            # probably a triple intersection
                            isInside = insideTriangle(sensors, [k, sensor.neighbours[j].nid,
                                                                sensor.neighbours[i].nid], intrsxn)
                            if isInside == True:
                                # sensor.neighbours[i][a] = sensor.neighbours[i][a] + \
                                #     (sensor.neighbours,ni,)
                                break
                        # because float thats why
                        elif distsq < (sensors[sensor.neighbours[j].nid].r**2) + 0.0000000001:
                            isInside = True
                            # sensor.neighbours[i][a] = sensor.neighbours[i][a] + \
                            #     (sensor.neighbours[j][0][0],)
                            break
                if isInside == False:
                    sensor.boundaryLevel = 0
                    sensor.boundaryPoints.append(intrsxn)


def getBoundaryArc(sensors):
    for sensor in sensors:
        sensor.boundaryArc.clear()
    for i in range(0, len(sensors)):
        angles = []
        sensor = sensors[i]
        if sensor.boundaryLevel != -1:
            if len(sensor.boundaryPoints) < 2:
                sensor.boundaryPoints.clear()
                # debugPlot(sensors, i)
                # pudb.set_trace()
                sensor.boundaryLevel = -1
                continue
            for point in sensor.boundaryPoints:
                dx = point.x-sensor.x
                dy = sensor.y-point.y  # reverse y direction
                theta = degrees(atan2(dy, dx))
                if theta < 0:
                    theta = theta+360
                angles.append((theta, point))  # neighbour point
            angles = sorted(angles, key=lambda x: x[0])
            num = len(angles)
            medians = []
            for k in range(0, num):
                theta1 = angles[k][0]
                theta2 = angles[(k+1) % num][0]
                if theta1 < theta2:
                    mtheta = (theta1+theta2)/2
                elif theta1 > theta2:
                    mtheta = (theta1+theta2)/2 - 180
                    if mtheta < 0:
                        mtheta = mtheta+360
                else:
                    continue
                xm = sensor.x + sensor.r*cos(radians(mtheta))
                ym = sensor.y - sensor.r*sin(radians(mtheta))
                medians.append((xm, ym))
                # area.create_circle(xm*SF,ym*SF,SF,fill='pink')
                isBoundaryArc = True
                for j in range(0, len(sensor.neighbours)):
                    dx = abs(xm-sensors[sensor.neighbours[j].nid].x)
                    dy = abs(ym-sensors[sensor.neighbours[j].nid].y)
                    dist = sqrt(dx**2+dy**2)
                    coinc = []
                    circ = []
                    if abs(dist-sensors[sensor.neighbours[j].nid].r) < 0.0000000001:
                        ts = sensors[sensor.neighbours[j].nid]
                        if (ts.x, ts.y) not in coinc:
                            coinc.append((ts.x, ts.y))
                            circ.append(sensor.neighbours[j].nid)
                    elif dist < sensors[sensor.neighbours[j].nid].r:
                        isBoundaryArc = False
                        break
                if isBoundaryArc and len(circ) >= 2:
                    isInside = insideTriangle(
                        sensors, [i, circ[0], circ[1]], Point(xm, ym))
                    if isInside == True:
                        isBoundaryArc = False
                if isBoundaryArc:
                    sensor.boundaryArc.append((angles[k], angles[(k+1) % num]))
            # if len(sensor.boundaryArc) == 0:
            #     debugPlot(sensors, i, medians)
            #     pudb.set_trace()


def debugPlot(sensors, sensorNum, extraPoints=[]):
    area.delete('all')
    area.update()
    area.update_idletasks()
    for sensor in sensors:
        area.create_circle(sensor.x*SF, sensor.y*SF,
                           sensor.r*SF, outline="black", width=1)

    for x in sensors[sensorNum].neighbours:
        sensor = sensors[x[0][0]]
        area.create_circle(sensor.x*SF, sensor.y*SF,
                           sensor.r*SF, outline="red", width=1)
        area.create_text(sensor.x*SF, sensor.y*SF,
                         text=x[0][0].__str__(), fill="pink")
        for intersection in x:
            area.create_circle(
                intersection[1]*SF, intersection[2]*SF, SF/20, fill='red')
            if len(intersection) == 4:
                area.create_text(
                    intersection[1]*SF, intersection[2]*SF, text=intersection[3].__str__())
    sensor = sensors[sensorNum]
    area.create_circle(sensor.x*SF, sensor.y*SF, sensor.r *
                       SF, outline="green", width=1)
    for sb in sensor.boundaryPoints:
        area.create_circle(sb[1]*SF, sb[2]*SF, SF/2, outline="blue", width=1)
    for ep in extraPoints:
        area.create_circle(ep[0]*SF, ep[1]*SF, SF/5, fill='yellow')
    area.update()
    area.update_idletasks()


def drawPlot(sensors):
    for sensor in sensors:
        area.create_circle(sensor.x*SF, sensor.y*SF,
                           sensor.r*SF, outline="#000", width=1)
        if sensor.boundaryLevel != -1:
            for arcs in sensor.boundaryArc:
                arcl = arcs[1][0]-arcs[0][0]
                if (arcl < 0):
                    arcl = arcl+360
                area.create_arc((sensor.x-sensor.r)*SF,
                                (sensor.y-sensor.r)*SF,
                                (sensor.x+sensor.r)*SF,
                                (sensor.y+sensor.r)*SF,
                                start=arcs[0][0],
                                extent=arcl,
                                outline="blue", style=tk.ARC,
                                width=2)
            for intrsxn in sensor.maskedIntersections:
                area.create_circle(
                    intrsxn[0]*SF, intrsxn[1]*SF, SF/2, outline="blue", width=1)
    for intersection in intersections:
        area.create_circle(intersection.x*SF,
                           intersection.y*SF, SF/5, fill='red')


def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def getHoles(sensors):
    Holes.clear()
    visitedarcs = []
    for i in range(0, len(sensors)):
        if sensors[i].boundaryLevel != -1:
            HoleArcs = []
            for arcs in sensors[i].boundaryArc:
                if arcs not in visitedarcs:
                    visitedarcs.append(arcs)
                    initArc = arcs
                    p1 = initArc[0][1]
                    p = initArc[1][1]
                    sensorNum = i
                    currArc = initArc
                    HoleArcs.append((currArc, sensorNum, p))
                    while p != p1:
                        if (currArc, sensorNum, p) not in HoleArcs:
                            HoleArcs.append((currArc, sensorNum, p))
                        sensorNum = p.data
                        for arc in sensors[sensorNum].boundaryArc:
                            if arc[0][1] == p:
                                p = arc[1][1]
                                currArc = arc
                                visitedarcs.append(currArc)
                                break
                            elif arc[1][1] == p:
                                p = arc[0][1]
                                currArc = arc
                                visitedarcs.append(currArc)
                                break
                        # if no next sensor was found.... this should not occur
                        if sensorNum is p.data:
                            pudb.set_trace()
                    if (currArc, sensorNum, p) not in HoleArcs:
                        HoleArcs.append((currArc, sensorNum, p))
                if len(HoleArcs) != 0:
                    Holes.append(HoleArcs.copy(),)
                    HoleArcs.clear()


def getHoleArea(sensors, Holes):
    totalArea = 0
    for hole in Holes:
        sArea = 0
        for arcset in hole:
            arc = arcset[0]
            arcl = arc[1][0]-arc[0][0]
            if (arcl < 0):
                arcl = arcl+360
            sp = sensors[arcset[1]]
            # area of segment = area of sector - area of triangle for minor segment
            # area of segment = area of sector + area of triangle for major segment
            sArea = sArea + np.pi*sp.r*sp.r*arcl/360
            if arcl > 180:
                sArea = sArea + PolyArea([sp.x, arc[0][1].x, arc[1][1].y],
                                         [sp.y, arc[0][1].y, arc[1][1].y])
            else:
                sArea = sArea - PolyArea([sp.x, arc[0][1].x, arc[1][1].y],
                                         [sp.y, arc[0][1].y, arc[1][1].y])
        X = [i[2].x for i in hole]
        Y = [i[2].y for i in hole]
        pArea = PolyArea(X, Y)
        hArea = pArea-sArea
        if hArea < -0.01:
            raise Exception("area calculation error")
        totalArea = totalArea+hArea
    print("area of hole is "+totalArea.__str__())


def getApproxArea(sensors):
    partlyCovered = []
    unCovered = []
    for x in range(0, canvasX):
        for y in range(0, canvasY):
            isPartCovered = False
            isCovered = False
            for sensor in sensors:
                sx, sy, sr = sensor.x, sensor.y, sensor.r
                a = sqrt((sx-x)**2+(sy-y)**2) < sr
                b = sqrt((sx-(x+1))**2+(sy-y)**2) < sr
                c = sqrt((sx-x)**2+(sy-(y+1))**2) < sr
                d = sqrt((sx-(x+1))**2+(sy-(y+1))**2) < sr
                if (a and b and c and d):
                    isCovered = True
                    break
                elif(a or b or c or d):
                    isPartCovered = True
            if (isPartCovered == True and isCovered == False):
                partlyCovered.append((x, y))
            if (isCovered == False and isPartCovered == False):
                unCovered.append((x, y))
    area = len(unCovered)
    for a in partlyCovered:
        (x, y) = a
        for i in range(0, 10):
            for j in range(0, 10):
                px = x+i/10
                py = y+j/10
                isPartCovered = False
                isCovered = False
                for sensor in sensors:
                    sx, sy, sr = sensor.x, sensor.y, sensor.r
                    a = sqrt((sx-px)**2+(sy-py)**2) < sr
                    b = sqrt((sx-(px+1))**2+(sy-py)**2) < sr
                    c = sqrt((sx-px)**2+(sy-(py+1))**2) < sr
                    d = sqrt((sx-(px+1))**2+(sy-(py+1))**2) < sr
                    if (a and b and c and d):
                        isCovered = True
                        break
                    elif(a or b or c or d):
                        isPartCovered = True
                if (isPartCovered == True and isCovered == False):
                    area = area+0.004
                if (isCovered == False and isPartCovered == False):
                    area = area+0.01
    print("the approximated area is "+area.__str__())


def getMaskedIntersections(sensors):
    for i in range(0, len(sensors)):
        sensor = sensors[i]
        sensor.maskedIntersections.clear()
        if sensor.boundaryLevel != -1:
            # All masked intrsxns will be intersection of its neighbours
            for nb in sensor.neighbours:
                nbIndex = nb.nid
                ns = sensors[nbIndex]
                for k in range(0, len(ns.neighbours)):
                    for intrsxn in ns.neighbours[k].points:
                        isInside = False
                        dx = sensor.x - intrsxn.x
                        dy = sensor.y - intrsxn.y
                        dist = sqrt(dx**2 + dy**2)
                        # intersection is not masked by if its not covered by s
                        if dist > sensor.r:
                            continue
                        for j in range(0, len(ns.neighbours)):
                            snj = sensors[ns.neighbours[j].nid]
                            sj = sensors[j]
                            sk = sensors[k]
                            if (snj.x != sensor.x or snj.y != sensor.y) and (sj.x != sk.x or sj.y != sk.y):
                                dx = abs(
                                    intrsxn.x-sensors[ns.neighbours[j].nid].x)
                                dy = abs(
                                    intrsxn.y-sensors[ns.neighbours[j].nid].y)
                                dist = sqrt(dx**2+dy**2)
                                if dist < sensors[ns.neighbours[j].nid].r:
                                    if sensors[ns.neighbours[j].nid].boundaryLevel == -1:
                                        isInside = True
                                        break
                        if isInside == False:
                            if intrsxn not in sensor.maskedIntersections:
                                sensor.maskedIntersections.append(intrsxn)


# type 0 for our
def getnewcentre(sc, spr, sn, sensors, boundrySensors, stratType):
    scurr = sensors[boundrySensors[sc]]
    sprev = sensors[boundrySensors[spr]]
    if sn == len(boundrySensors):
        sn = 0
    snext = sensors[boundrySensors[sn]]
    ls=LineSegment(sprev.getCentre(),snext.getCentre())
    TC = []   # new tentative centres
    for p in scurr.maskedIntersections:
        
        if p in scurr.boundaryPoints:
            continue
        if stratType == 0:
                pts = getLineCircleIntersection(ls.getLineObject(),scurr.getCentre(),scurr.r)

        elif stratType == 1:
            arc = scurr.boundaryArc[0]
            arcl = arc[1][0]-arc[0][0]
            if (arcl < 0):
                arcl = arcl+360
            m = (arc[0][0]+(random.random()*arcl)) % 360
            m = 360-m  # take mirror image of arc because tkinter
            pts = getLineCircleIntersection(Line(p,m),scurr.getCentre,scurr.r)
        else:
            pts = None
        if pts == None:
            return (scurr.x, scurr.y)
        for pt in pts:
            if abs(pt[0]-scurr.x) < 0 and abs(pt[1]-scurr.y) < 0:
                continue
            agl = degrees(atan2(scurr.y-pt[1], pt[0]-scurr.x))
            if agl < 0:
                agl = agl+360

            if abs(agl - (scurr.boundaryArc[0][1][0] + scurr.boundaryArc[0][0][0])/2) < 90:
                TC.append(pt)
    mindist = 10000000
    minpos = (sensors[boundrySensors[0]].x, sensors[boundrySensors[0]].y)
    for p in TC:
        dist = sqrt((scurr.x-p[0])**2+(scurr.y-p[1])**2)
        if dist < mindist:
            mindist = dist
            minpos = p
    return (minpos)


def Saveplot(imageName):
    ps = area.postscript(colormode='color')
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img.save(imageName, 'PNG')

def healHole(sensors,Hole):
    hexagons=scatterHexagons()
    borderHexes=[]
    for hexagon in hexagons:
        isborder=hexagonOnHole(hexagon,Hole)
        if isborder:
            plotHexagon(hexagon)
            borderHexes.append(hexagon)
    Saveplot('hexident')
    for i in range(min(len(Hole),len(borderHexes))):      
        sensors[Hole[i][1]].x=borderHexes[i].centre.x
        sensors[Hole[i][1]].y=borderHexes[i].centre.y
    return borderHexes


def shrinkHole(sensors, Hole, stratType):
    boundrySensors = [arc[1] for arc in Hole]
    print(boundrySensors)
    l = len(boundrySensors)
    if l % 2 == 1:
        for i in range(1, len(boundrySensors), 2):
            # also send the arc for random case
            tmp = (getnewcentre(i, i-1, (i+1) %
                                l, sensors, boundrySensors, stratType))
            sensors[boundrySensors[i]].x = tmp[0]
            sensors[boundrySensors[i]].y = tmp[1]
            getHoleBoundary(sensors)
            getBoundaryArc(sensors)
            getMaskedIntersections(sensors)
        for i in range(2, len(boundrySensors), 2):
            tmp = (getnewcentre(i, i-1, (i+1) %
                                l, sensors, boundrySensors, stratType))
            sensors[boundrySensors[i]].x = tmp[0]
            sensors[boundrySensors[i]].y = tmp[1]
            getHoleBoundary(sensors)
            getBoundaryArc(sensors)
            getMaskedIntersections(sensors)
        c = getnewcentre(0, -1, 1, sensors, boundrySensors, stratType)
        sensors[boundrySensors[0]].x = c[0]
        sensors[boundrySensors[0]].y = c[1]
        getHoleBoundary(sensors)
        getBoundaryArc(sensors)
        getMaskedIntersections(sensors)
    else:
        for i in range(0, len(boundrySensors), 2):
            tmp = (getnewcentre(i, i-1, (i+1) %
                                l, sensors, boundrySensors, stratType))
            sensors[boundrySensors[i]].x = tmp[0]
            sensors[boundrySensors[i]].y = tmp[1]
            getHoleBoundary(sensors)
            getBoundaryArc(sensors)
            getMaskedIntersections(sensors)
        for i in range(1, len(boundrySensors), 2):
            tmp = (getnewcentre(i, i-1, (i+1) %
                                l, sensors, boundrySensors, stratType))
            sensors[boundrySensors[i]].x = tmp[0]
            sensors[boundrySensors[i]].y = tmp[1]
            getHoleBoundary(sensors)
            getBoundaryArc(sensors)
            getMaskedIntersections(sensors)


def ShrinkOne(sensors, Hole, stratType):
    boundrySensors = [arc[1] for arc in Hole]
    c = getnewcentre(0, -1, 1, sensors, boundrySensors, stratType)
    sensors[boundrySensors[0]].x = c[0]
    sensors[boundrySensors[0]].y = c[1]
    getHoleBoundary(sensors)
    getBoundaryArc(sensors)
    getMaskedIntersections(sensors)


def ShowSteps(sensors):
    for sensor in sensors:
        area.create_circle(sensor.x*SF, sensor.y*SF,
                           sensor.r*SF, outline="#000", width=1)
    Saveplot('sensdist.png')
    flag = True
    for intersection in intersections:
        area.create_circle(intersection[0]*SF,
                           intersection[1]*SF, SF/5, fill='red')
        if flag:
            Saveplot('first_inter.png')
            flag = False
    Saveplot('all_inter.png')
    flag = True
    flag2 = True
    for sensor in sensors:
        if sensor.isBoundry == True:
            for arcs in sensor.boundaryArc:
                arcl = arcs[1][0]-arcs[0][0]
                area.create_circle(
                    arcs[1][1][1]*SF, arcs[1][1][2]*SF, SF/5, fill='blue')
                if flag:
                    Saveplot('boundry_first.png')
                area.create_circle(
                    arcs[0][1][1]*SF, arcs[0][1][2]*SF, SF/5, fill='blue')
                if flag:
                    Saveplot('boundry_first.png')
                if (arcl < 0):
                    arcl = arcl+360
                area.create_arc((sensor.x-sensor.r)*SF,
                                (sensor.y-sensor.r)*SF,
                                (sensor.x+sensor.r)*SF,
                                (sensor.y+sensor.r)*SF,
                                start=arcs[0][0],
                                extent=arcl,
                                outline="blue", style=tk.ARC,
                                width=2)
                if flag:
                    Saveplot('arc_first.png')
                    flag = False
            Saveplot('arc_all.png')
            for intrsxn in sensor.maskedIntersections:
                area.create_circle(
                    intrsxn[0]*SF, intrsxn[1]*SF, SF/2, outline="blue", width=1)
                if flag2:
                    Saveplot('extreme_first.png')
                    flag2 = False
                Saveplot('extreme_all.png')
    area.delete('all')
    area.update()
    area.update_idletasks()
    ShrinkOne(sensors, Holes[0], stratType=0)
    drawPlot(sensors)
    Saveplot('Movement_first.png')
    area.delete('all')
    area.update()
    area.update_idletasks()
    shrinkHole(sensors, Holes[0], stratType=0)
    drawPlot(sensors)
    Saveplot('Movement_All.png')


def repat(sensors,strattype):
    Saveplot("before")
    if strattype=='opt':
        shrinkHole(sensors, Holes[0], stratType=0)
    elif strattype=='random':
        shrinkHole(sensors, Holes[0], stratType=0)
    elif strattype=='heal':
        H=healHole(sensors,Holes[0])
    else :
        raise Exception("Unknown recovery Stratergy")
    area.delete('all')
    area.update()
    area.update_idletasks()
    if strattype=='heal':
        for h in H:
            plotHexagon(h)
    for sensor in sensors:
        sensor.clear_sensor_data()
    getHoleBoundary(sensors)
    getBoundaryArc(sensors)
    getHoles(sensors)
    # getMaskedIntersections(sensors)
    drawPlot(sensors)
    # getHoleArea(sensors, Holes)
    Saveplot("after")


def getCombinedPolygons(Hexagons):
    poly_lines = []
    Hexagons_l = [x.getLineSegments() for x in Hexagons]
    for hexagon_l in Hexagons_l:
        for line in hexagon_l:
            if line in poly_lines:
                poly_lines.remove(line)
            else:
                poly_lines.append(line)
    return poly_lines

def getSensorPolygonIntersections(sensor,polygon):
    lineSegments=polygon.getLineSegments()
    res=[]
    for ls in lineSegments:
        res=res+getLineSegmentSensorIntersection(ls,sensor)
    return res
    
    
def plotHexagon(hexagon):
    area.create_polygon(*[(SF*x.x, SF*x.y)
                          for x in hexagon], outline="blue", fill="")


def scatterHexagons():
    hexagons = []
    for i in range((canvasX//radius)):
        if i % 2 == 0:
            centre = Point(0, radius*i*(1.5))
        else:
            centre = Point((sqrt(3)/2)*radius, radius*i*1.5)
        for j in range((canvasX//radius)):
            hexagons.append(RegularHexagon(centre, radius))
            centre.x = centre.x+radius*sqrt(3)
    return hexagons

def hexagonOnHole(hexagon,Hole):
    for holeArc in Hole:
        s=msensors[holeArc[1]]
        i=getSensorPolygonIntersections(s,hexagon)
        for point in i:
            for arc in s.boundaryArc:
                arcl = arc[1][0]-arc[0][0]# thetas of 
                if (arcl < 0):
                    arcl = arcl+360
                dx = point.x-s.x
                dy = s.y-point.y  # reverse y direction
                theta = degrees(atan2(dy, dx))
                if theta < 0:
                    theta = theta+360
                arcl2=theta-arc[0][0]
                if (arcl2 < 0):
                    arcl2 = arcl2+360
                if arcl2 <= arcl:
                    return True
    return False

ScatterSensors(msensors)
gridx = canvasX*SF
gridy = canvasY*SF

window = tk.Tk()
window.title("SenSim")
area = tk.Canvas(window, height=gridy, width=gridx)
area.configure(background='white')
area.pack()
xp = 25+int(random.random()*50)
yp = 25+int(random.random()*50)

tsensors = []
for sensor in msensors:
    dx = abs(sensor.x-xp)
    dy = abs(sensor.y-yp)
    dist = sqrt(dx**2+dy**2)
    if dist >= sensor.r+4:
        tsensors.append(sensor)
msensors.clear()
msensors = tsensors
getHoleBoundary(msensors)

getBoundaryArc(msensors)
getHoles(msensors)
getHoleArea(msensors, Holes)
getMaskedIntersections(msensors)
# sensor=Sensor(Point(xp,yp),radius)
# ls=LineSegment(Point(xp-20,yp-20),Point(xp+20,yp+15))
# intersections+=getLineSegmentSensorIntersection(ls,sensor)

   
drawPlot(msensors)
# getApproxArea(msensors)
area.after(2000,lambda : repat(msensors,'heal'))
window.mainloop()
