import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

'''
Data links:
The bedrock: https://sites.uci.edu/morlighem/dataproducts/bedmachine-greenland/
surface data: https://nsidc.org/data/measures/gimp
precipitation: https://www.projects.science.uu.nl/iceclimate/models/greenland.php

Tutorials/Documentation:
https://pyhogs.github.io/intro_netcdf4.html
https://semba-blog.netlify.app/07/04/2020/mapping-with-cartopy-in-python/
'''

f = nc.Dataset('/home/stefano/Scaricati/BedMachineGreenland-2021-04-20.nc')
step_size = 10**1 # too big dataset, can't handle all points on laptop
x = f.variables['x'][::step_size]
y = f.variables['y'][::step_size]
xd, yd = np.meshgrid(x,y)

## LAND TYPE
mask = f.variables['mask'][::step_size, ::step_size]
plt.figure()
pl = plt.contourf(xd, yd, mask, cmap='jet', levels=[-1,0,1,2,3,4])
plt.title('Visualization of BedMachineGreenland-2021-04-20.nc (land types)')
labels = f.variables['mask'].flag_meanings.split()
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
     for pc in pl.collections]
plt.legend(proxy, labels)
plt.show()
