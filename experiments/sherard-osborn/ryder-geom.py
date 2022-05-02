import matplotlib.pyplot as plt
from fenics import near

x, bed, ice_thickness, ice_surface, ice_profile = [], [], [], [], []

input = open('RyderLonger_dx200m_0001_Input.txt', 'r')
data = input.readlines()
for line in data:
    values = line.split(' ')

    # before x = 280k it is ice only; grounding line is at ~285k
    # if ice profile is within 5 meters from bed, it means we still haven't reached grounding line
    if float(values[3])-float(values[2]) - float(values[1]) < 5:
        continue
        
    x.append(float(values[0]))
    bed.append(float(values[1]))
    ice_thickness.append(float(values[2]))
    ice_surface.append(float(values[3]))
    ice_profile.append(float(values[3])-float(values[2]))

# rescale so that origin is at (0,0)
min_x = min(x)
min_bed = min(bed)
min_ice_profile = min(ice_profile)
x = [el-min_x for el in x]
bed = [el-min_bed for el in bed]
ice_profile = [el-min_bed for el in ice_profile]
# find entry when ice profile becomes flat and cut list there
for idx in range(len(ice_profile)):
    if x[idx] > 28000 and near(ice_profile[idx], ice_profile[idx+1]):
        ice_profile = ice_profile[0:idx]
        break

plt.scatter(x,bed)
#plt.scatter(x,ice_thickness)
#plt.scatter(x,ice_surface)
plt.scatter(x[0:len(ice_profile)],ice_profile)
#plt.legend(['bed','ice thickness','ice surface','ice_profile'])
plt.legend(['bed','ice_profile'])
plt.show()
