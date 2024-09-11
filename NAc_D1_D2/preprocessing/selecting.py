import pickle
from caiman.base.rois import register_multisession
from caiman.base.rois import register_ROIs
from caiman.utils import visualization
from caiman.utils.utils import download_demo
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'
from matplotlib.widgets import Button

file_name = "MC"

with open('{}.tiff_cn_filter.txt'.format(file_name), 'rb') as fp:
    cn_filter = pickle.load(fp) 
dims = cn_filter.shape

with open('{}.tiff_A.txt'.format(file_name), 'rb') as fp:
    spatial = pickle.load(fp)

with open('{}.tiff_cn_filter_zero.txt'.format(file_name), 'rb') as fp:
    cn_zero = pickle.load(fp) 


with open('{}.tiff_DF.txt'.format(file_name), 'rb') as fp:
    denoised_z_traces = pickle.load(fp) 


from caiman.utils.visualization import get_contours

maxthr=0.2
nrgthr=0.9
thr_method = 'max'

thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
#idx=cnm.estimates.idx_components

coordinates = get_contours(spatial, dims, thr, thr_method)

with open('MC'+file_name+'.tiff.txt') as f:
    fname_new = f.readlines()
fname_new = fname_new[0]
d1 = int(fname_new[11:14])
d2 = int(fname_new[18:21])

gSiz = [13,13]

def norm(a, rg=(0, 1)):
    a_norm = (a - a.min()) / (a.max() - a.min())
    return a_norm * (rg[1] - rg[0]) + rg[0]

import matplotlib as mpl
import matplotlib.cm as cmm

A = spatial

Ad = np.asarray(A.todense()).reshape((d1, d2, -1), order='F')
myfoot = []
for i in range(spatial.shape[1]):
    A_scl = norm(Ad[:, :, i], (0, 1))
#     smp = cmm.ScalarMappable(cmap='gray')
#     im_rgb = smp.to_rgba(cn_filter)[:, :, :3]
#     im_hsv = mpl.colors.rgb_to_hsv(im_rgb)
#     im_hsv_scl = im_hsv.copy()
#     im_hsv_scl[:, :, 2] = im_hsv[:, :, 2] * A_scl
    myfoot.append(A_scl)
myfoot = np.array(myfoot)

footprints = myfoot
colormap='gist_rainbow'
composite_fov = np.zeros((footprints.shape[1], footprints.shape[2], 3))
cmap_vals = cmm.get_cmap(colormap)
np.random.seed(0)
for cell_id in range(footprints.shape[0]):
    # select a random color for this cell
    color = cmap_vals(np.random.rand())

    # assign the color to each of the three channels, normalized by the footprint peak
    for color_channel in range(3):
        composite_fov[:,:,color_channel] += color[color_channel]*footprints[cell_id]/np.max(footprints[cell_id])

# set all values > 1 (where cells overlap) to 1:
composite_fov[np.where(composite_fov > 1)] = 1

##############################################################################################
plt.rcParams['figure.facecolor'] = 'white'
import os
import shutil

path=file_name+"_accepted"

try:
    if os.path.exists(path):
        for k in range(5):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ATTENTION! Result folders already exist and will be deleted if you proceed. To stop the process, close the program.")
        input("Press enter to continue")
        shutil.rmtree(path)
        path=file_name+"_rejected"
        shutil.rmtree(path)

    path=file_name+"_rejected"
    if os.path.exists(path):
        for k in range(5):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ATTENTION! Result folders already exist and will be deleted if you proceed. To stop the process, close the program.")
        input("Press enter to continue")
        shutil.rmtree(path)
        
    path=file_name+"_maybe"
    if os.path.exists(path):
        for k in range(5):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ATTENTION! Result folders already exist and will be deleted if you proceed. To stop the process, close the program.")
        input("Press enter to continue")
        shutil.rmtree(path)
        
except:
    print("-------------------------")


folder= file_name+"_accepted"
try:
    os.mkdir(folder)
except:
    print('Folder ok')
    
folder=file_name+"_rejected"
try:
    os.mkdir(folder)
except:
    print('Folder ok')
    
folder=file_name+"_maybe"
try:
    os.mkdir(folder)
except:
    print('Folder ok')

i = 0

f, axs = plt.subplots(2, 4, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12,8))
#a0.imshow(np.reshape(cnm.estimates.A[:,cnm.estimates.idx_components[i]].toarray(), dims, order='F'),cmap='gray')

a0 = axs[0,0]
a0.set_title('Correlation', fontsize=14)

a0.imshow(cn_filter, cmap='viridis', interpolation='nearest')

c = coordinates[i]
v = c['coordinates']
c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
             np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
a0.plot(*v.T, c='white', lw=2)

#a0.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

ax = axs[0,1]
ax.set_title('Zoom', fontsize=14)


ax.imshow(cn_filter, cmap='viridis', interpolation='nearest')

c = coordinates[i]
v = c['coordinates']
c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
             np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
ax.plot(*v.T, c='white', lw=2)

ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])



ax = axs[0,2]
ax.set_title('No filter', fontsize=14)

ax.imshow(cn_zero, cmap='jet', interpolation='nearest')

c = coordinates[i]
v = c['coordinates']
c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
             np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
ax.plot(*v.T, c='white', lw=2)

ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])

ax = axs[0,3]
ax.set_title('Lights', fontsize=14)
for cell_id in range(footprints.shape[0]):
    peak_loc = np.where(footprints[cell_id]==np.max(footprints[cell_id]))
    ax.text(
        peak_loc[1][0], 
        peak_loc[0][0], 
        'cell {}'.format(str(cell_id).zfill(2)), 
        color='white', 
        ha='center', 
        va='center',
        fontweight='bold',clip_on=True
    )

# show the image
ax.imshow(composite_fov)
ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])



gs = axs[1,0].get_gridspec() #juntados...
# remove the underlying axes
for ax in axs[1,:]:
    ax.remove()
a1 = f.add_subplot(gs[1,:]) #plotar no grafico juntados
a1.set_title('Trace', fontsize=14)
a1.set_xlabel("Time (s)", fontsize=14)
a1.set_ylabel("DF/F", fontsize=14)
a1.set_xlim(0,len(denoised_z_traces[i])*50/1000)
a1.axhline(y=5, color="orange", ls="--")

#a1 = axs[1,0]
#a1.plot(cnm.estimates.C[cnm.estimates.idx_components[i]])
a1.plot(np.arange(0,len(denoised_z_traces[i]),1)*50/1000,denoised_z_traces[i])


f.suptitle("Accepted Components ID {}".format(i), fontsize=16)
f.tight_layout()
f.subplots_adjust(top=0.95)
f.patch.set_facecolor('white')
#plt.savefig("{}/ID_{:03d}.png".format(fnames[0][:-5],i), dpi=400)

select_id = np.ones(spatial.shape[1])
max_id = spatial.shape[1]

def accept(event):
    global i, a1, max_id, f,file_name
    plt.savefig("{}/ID_{:03d}.png".format(file_name+"_accepted",i), dpi=100)

    i+=1
    if (i==max_id):
        plt.close()
        print("Data saved!")
        print("The program can be closed.")
        #fim()
        from sys import exit
        exit()
        return ("THE END!")
        
        
    f.suptitle("Accepted Components ID {}".format(i), fontsize=16)

    a0 = axs[0,0]
    a0.clear()
    
    a0 = axs[0,0]
    a0.set_title('Correlation', fontsize=14)

    a0.imshow(cn_filter, cmap='viridis', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    a0.plot(*v.T, c='white', lw=2)
    
    ax = axs[0,1]
    ax.clear()
    ax.set_title('Zoom', fontsize=14)


    ax.imshow(cn_filter, cmap='viridis', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    ax.plot(*v.T, c='white', lw=2)

    ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])
    
    
    ax = axs[0,2]
    ax.clear()
    ax.set_title('No filter', fontsize=14)

    ax.imshow(cn_zero, cmap='jet', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    ax.plot(*v.T, c='white', lw=2)

    ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])
    
    ax = axs[0,3]
    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])

    
    #a1 = f.add_subplot(gs[1,:]) #plotar no grafico juntados
    a1.clear()
    a1.set_title('Trace', fontsize=14)
    a1.set_xlabel("Time (s)", fontsize=14)
    a1.set_ylabel("DF/F", fontsize=14)
    a1.set_xlim(0,len(denoised_z_traces[i])*50/1000)
    a1.axhline(y=5, color="orange", ls="--")
    a1.plot(np.arange(0,len(denoised_z_traces[i]),1)*50/1000,denoised_z_traces[i])
    
def delete(event):
    global i, a1, max_id, f, file_name, select_id
    plt.savefig("{}/ID_{:03d}.png".format(file_name+"_rejected",i), dpi=100)

    select_id[i] = 0
    i+=1
    
    if (i==max_id):
        plt.close()
        print("Data saved!")
        print("The program can be closed.")
        #fim()
        from sys import exit
        exit()
        return ("THE END!")
    
    f.suptitle("Accepted Components ID {}".format(i), fontsize=16)
    
    a0 = axs[0,0]
    a0.clear()
    
    a0 = axs[0,0]
    a0.set_title('Correlation', fontsize=14)

    a0.imshow(cn_filter, cmap='viridis', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    a0.plot(*v.T, c='white', lw=2)
    
    ax = axs[0,1]
    ax.clear()
    ax.set_title('Zoom', fontsize=14)


    ax.imshow(cn_filter, cmap='viridis', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    ax.plot(*v.T, c='white', lw=2)

    ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])
    
    
    ax = axs[0,2]
    ax.clear()
    ax.set_title('No filter', fontsize=14)

    ax.imshow(cn_zero, cmap='jet', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    ax.plot(*v.T, c='white', lw=2)

    ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])
    
    ax = axs[0,3]
    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])

    
    #a1 = f.add_subplot(gs[1,:]) #plotar no grafico juntados
    a1.clear()
    a1.set_title('Trace', fontsize=14)
    a1.set_xlabel("Time (s)", fontsize=14)
    a1.set_ylabel("DF/F", fontsize=14)
    a1.set_xlim(0,len(denoised_z_traces[i])*50/1000)
    a1.axhline(y=5, color="orange", ls="--")
    a1.plot(np.arange(0,len(denoised_z_traces[i]),1)*50/1000,denoised_z_traces[i])
    
    
    
def maybe(event):
    global i, a1, max_id, f, file_name, select_id
    plt.savefig("{}/ID_{:03d}.png".format(file_name+"_maybe",i), dpi=100)

    select_id[i] = 0
    i+=1
    
    if (i==max_id):
        plt.close()
        print("Data saved!")
        print("The program can be closed.")
        #fim()
        from sys import exit
        exit()
        return ("THE END!")
    
    f.suptitle("Accepted Components ID {}".format(i), fontsize=16)
    
    a0 = axs[0,0]
    a0.clear()
    
    a0 = axs[0,0]
    a0.set_title('Correlation', fontsize=14)

    a0.imshow(cn_filter, cmap='viridis', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    a0.plot(*v.T, c='white', lw=2)
    
    ax = axs[0,1]
    ax.clear()
    ax.set_title('Zoom', fontsize=14)


    ax.imshow(cn_filter, cmap='viridis', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    ax.plot(*v.T, c='white', lw=2)

    ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])
    
    
    ax = axs[0,2]
    ax.clear()
    ax.set_title('No filter', fontsize=14)

    ax.imshow(cn_zero, cmap='jet', interpolation='nearest')

    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    ax.plot(*v.T, c='white', lw=2)

    ax.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2)

    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])
    
    ax = axs[0,3]
    ax.set_ylim(c["CoM"][0]+gSiz[0],c["CoM"][0]-gSiz[0])
    ax.set_xlim(c["CoM"][1]-gSiz[0],c["CoM"][1]+gSiz[0])

    
    #a1 = f.add_subplot(gs[1,:]) #plotar no grafico juntados
    a1.clear()
    a1.set_title('Trace', fontsize=14)
    a1.set_xlabel("Time (s)", fontsize=14)
    a1.set_ylabel("DF/F", fontsize=14)
    a1.set_xlim(0,len(denoised_z_traces[i])*50/1000)
    a1.axhline(y=5, color="orange", ls="--")
    a1.plot(np.arange(0,len(denoised_z_traces[i]),1)*50/1000,denoised_z_traces[i])
    
    
# Home button
axButn1 = plt.axes([0.72, 0.90, 0.07, 0.05])
btn1 = Button(
  axButn1, label="Accept", hovercolor='green')

btn1.on_clicked(accept)


# Previous button
axButn2 = plt.axes([0.88, 0.90, 0.07, 0.05])
btn2 = Button(
  axButn2, label="Delete", hovercolor='red')

# Previous button
axButn3 = plt.axes([0.80, 0.90, 0.07, 0.05])
btn3 = Button(
  axButn3, label="Maybe", hovercolor='skyblue')

# # Previous button
# axButn2 = plt.axes([0.87, 0.92, 0.1, 0.05])
# btn2 = Button(
#   axButn2, label="Delete", hovercolor='red')

btn2.on_clicked(delete)

btn3.on_clicked(maybe)

#plt.tight_layout()
#plt.show()
plt.show(block=False)
plt.pause(214748)
plt.close()
print(select_id)