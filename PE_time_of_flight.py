import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'C:\\PycharmProjects\\THzProcClass3')
from THzData import THzData

tvl_file = 'C:\\Work\\Signal Modeling\\THz Data\\HDPE Lens\\Smooth Side (res=0.5mm, OD=60ps).tvl'

data = THzData(tvl_file)

# diameter of the PE disk in mm
diameter = 32  # mm
radius = diameter / 2

radius /= data.x_res  # convert to indices

center = np.array([data.y_step//2, data.x_step//2])

data.make_time_of_flight_c_scan()

for i in range(data.y_step):
    for j in range(data.x_step):
        if np.sqrt((i - center[0])**2 + (j - center[1])**2) > radius:
            data.tof_c_scan[i, j] = 0.0

plt.figure('Time of Flight C-Scan')
plt.imshow(data.tof_c_scan, interpolation='none', extent=data.c_scan_extent, cmap='jet',
           vmax=11.5, vmin=7.56)
plt.title('Time of Flight (ps)')
plt.xlabel('X Scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.grid()
plt.colorbar()

plt.figure('Raw C-Scan')
plt.imshow(data.c_scan, interpolation='none', cmap='gray')
plt.title('Pk to Pk Amplitude')
plt.xlabel('X Scan Location (mm)')
plt.ylabel('Y Scan Location (mm)')
plt.grid()
plt.colorbar()

plt.show()
