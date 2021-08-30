import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import math
from fenics import Point, plot
import mshr

'''
Data links:
https://dataservices.gfz-potsdam.de/pik/showshort.php?id=escidoc:1487953

Tutorials/Documentation:
https://pyhogs.github.io/intro_netcdf4.html
'''

f = nc.Dataset('/home/stefano/git/fenics-phd/Ocean1_input_geom_v1.01.nc')
x = f.variables['x']
y = f.variables['y']
bedrock = f.variables['bedrockTopography']

def angle_counterclockwise_sort(point):
    # Center point
    vector = [point[0]-origin[0], point[1]-origin[1]]
    lenvector = np.linalg.norm(vector)

    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0

    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    angle = math.atan2(normalized[1], normalized[0])

    # Negative angles represent points in y<0 plane, add 2pi to them
    if angle < 0:
        angle += 2*math.pi

    # Angle is first sorting criterium, then vector length
    return angle, lenvector

def create_layer(lower_h, higher_h):
    # if bedrock(x,y) <= height, include it
    pts = []
    for p_x in x[::]:
        for p_y in y[::]:
            if lower_h <= bedrock[np.where(np.asarray(y)==p_y)[0][0], np.where(np.asarray(x)==p_x)[0][0]] <= higher_h:
                pts.append((p_x, p_y))

    #return np.asarray(pts)
    return pts

#l = create_layer(-720)
#print(l)
#plt.scatter(l[:,0], l[:,1])
#plt.show()

span = min(np.asarray(bedrock).flatten())
span_prev = span - 10
n = 0
origin = (np.average(x), np.average(y))
while span < -715:#max(np.asarray(bedrock).flatten()):
    l = create_layer(span_prev, span)
    print('Created layer {} with {} points at height {}'.format(n, len(l), span))

    old_l = []
    while old_l != l:
        old_l = list(l)
        idx = 1
        l_sfrondato = [l[0]]
        while idx < len(l)-1:
            print(np.subtract(l[idx+1], l[idx-1]))
            if np.subtract(l[idx+1], l[idx-1])[0] != 0 and np.subtract(l[idx+1], l[idx-1])[1] != 0:
                print('adding point {}'.format(l[idx]))
                l_sfrondato.append(l[idx])
            idx += 1
        l_sfrondato.append(l[-1])
        print('sfrondato layer, ora {} punti'.format(len(l_sfrondato)))
        print(l_sfrondato)

        plt.scatter(np.asarray(l_sfrondato)[:,0], np.asarray(l_sfrondato)[:,1])
        plt.show()

        l = sorted(l_sfrondato, key=angle_counterclockwise_sort)
        print('sorted points')
        print(l)

    Points = [Point(p) for p in l_sfrondato]
    g2d = mshr.Polygon(Points)
    print('built polygon')
    g3d = mshr.Extrude2D(g2d, 1)#(span_prev-span)/1000) #height is in meters
    print('extruded 3d')

    try:
        domain += g3d
    except NameError:
        domain = g3d

    span += 10 #step size is 10m
    n += 1

m = mshr.generate_mesh(domain, 10)
print('generated mesh')
plot(m)
plt.show()