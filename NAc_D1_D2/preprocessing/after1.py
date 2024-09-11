fnames = ['MC.tiff']
import matplotlib
matplotlib.use('Agg')
import os
# try:
#     get_ipython().magic(u'load_ext autoreload')
#     get_ipython().magic(u'autoreload 2')
#     get_ipython().magic(u'matplotlib qt')
# except:
#     pass

import logging
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'

import numpy as np

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2

try:
    cv2.setNumThreads(0)
except:
    pass
import bokeh.plotting as bpl
import holoviews as hv
bpl.output_notebook()
hv.notebook_extension('bokeh')


# dataset dependent parameters
frate = 20                       # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds

# motion correction parameters
###################################
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
max_shifts = (5, 5)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)      # overlap between pathes (size of patch strides+overlaps)
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries

mc_dict = {
    'fnames': fnames,
    'fr': frate,
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan
}

opts = params.CNMFParams(params_dict=mc_dict)

motion_correct = False
#bord_px = 0
if motion_correct:
    # do motion correction rigid
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        #plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        #plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        #plt.legend(['x shifts', 'y shifts'])
        #plt.xlabel('frames')
        #plt.ylabel('pixels')

    bord_px = 0 if border_nan == 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=bord_px)
    data_save = open('MC'+fnames[0]+'.txt','w+')
    data_save.write(str(fname_new))
    data_save.close()
    os.system('rm -r {}'.format(fname_mc[0]))
    os.system('Del {}'.format(fname_mc[0]))

else:
    print('ok, no MC done.')
    bord_px = 0
    with open('MC'+fnames[0]+'.txt') as f:
        fname_new = f.readlines()
    fname_new = fname_new[0]
    # if no motion correction just memory map the file
    #fname_new = cm.save_memmap(fnames, base_name='memmap_',order='C', border_to_0=bord_px, dview=dview)

d1 = int(fname_new[11:14])
d2 = int(fname_new[18:21])

# load memory mappable file
Yr, dims, T = cm.load_memmap(fname_new)
images = Yr.T.reshape((T,) + dims, order='F')
# parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None
gSig = (5, 5)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (10, 10)     # average diameter of a neuron, in general 4*gSig+1
Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .995      # merging threshold, max correlation allowed
rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 1            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .7       # min peak value from correlation image
min_pnr = 5        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': bord_px})                # number of pixels to not consider in the borders)

#%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
plt.rcParams['figure.facecolor'] = 'white'

if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

cn_zero, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=0, swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile

cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile

from os.path import exists

file_exists = exists('CNMFE_'+fnames[0][:-5]+".hdf5")
if file_exists:
    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    cnm = load_CNMF('CNMFE_'+fnames[0][:-5]+".hdf5", n_processes=n_processes, dview=dview)
else:
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)
    cnm.save('CNMFE_'+fnames[0][:-5]+".hdf5")

#%% COMPONENT EVALUATION
# the components are evaluated in two ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
# Note that here we do not use the CNN based classifier, because it was trained on 2p not 1p data

min_SNR = 3            # adaptive way to set threshold on the transient size
r_values_min = 0.85    # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))

from scipy.signal import welch
from caiman.source_extraction.cnmf.deconvolution import GetSn

def zscore_trace(denoised_trace,
                 raw_trace,
                 offset_method = 'floor',
                 sn_method='logmexp',
                 range_ff=[0.25, 0.5]):
    """
    "Z-score" calcium traces based on a calculation of noise from
    power spectral density high frequency.

    Inputs:
        denoised_trace: from estimates.C
        raw_trace: from estimates.C + estimates.YrA
        offset_method: offset to use when shifting the trace (default: 'floor')
            'floor': minimum of the trace so new min will be zero
            'mean': mean of the raw/denoised trace as the zeroing point
            'median': median of raw/denoised trace
            'none': no offset just normalize
        sn_method: how are psd central values caluclated
            mean
            median' or 'logmexp')
        range_ff: 2-elt array-like range of frequencies (input for GetSn) (default [0.25, 0.5])

    Returns
        z_denoised: same shape as denoised_trace
        z_raw: same shape as raw trace
        trace_noise: noise level from z_raw

    Adapted from code by Zach Barry.
    """
    noise = GetSn(raw_trace, range_ff=range_ff, method=sn_method)  #import this from caiman

    if offset_method == 'floor':
        raw_offset = np.min(raw_trace)
        denoised_offset = np.min(denoised_trace)
    elif offset_method == 'mean':
        raw_offset = np.mean(raw_trace)
        denoised_offset = np.mean(denoised_trace)
    elif offset_method == 'median':
        raw_offset = np.median(raw_trace)
        denoised_offset = np.median(denoised_trace)
    elif offset_method == 'none':
        raw_offset = 0
        denoised_offset = 0
    else:
        raise ValueError("offset_method should be floor, mean, median, or none.")

    z_raw = (raw_trace - raw_offset) / noise
    z_denoised = (denoised_trace - denoised_offset)/ noise

    return z_denoised, z_raw, noise

print('loaded')


def zscore_traces(cnm_c,
                  cnm_yra,
                  offset_method = 'floor',
                  sn_method = 'logmexp',
                  range_ff=[0.25, 0.5]):
    """
    apply zscore_trace to all traces in estimates

    inputs:
        cnm_c: C array of denoised traces from cnm.estimates
        cnm_yra: YrA array of residuals from cnm.estimate
        offset_method: floor/mean/median (see zscore_trace)
        sn_method: mean/median/logmexp (see zscore_trace)
        range_ff: frequency range for GetSn

    outputs:
        denoised_z_traces
        raw_z_traces
        noise_all
    """
    raw_traces = cnm_c + cnm_yra  # raw_trace[i] = c[i] + yra[i]
    raw_z_traces = []
    denoised_z_traces = []
    noise_all = []
    for ind, raw_trace in enumerate(raw_traces):
        denoised_trace = cnm_c[ind,:]
        z_denoised, z_raw, noise = zscore_trace(denoised_trace,
                                                raw_trace,
                                                offset_method=offset_method,
                                                sn_method = sn_method,
                                                range_ff=range_ff)

        denoised_z_traces.append(z_denoised)
        raw_z_traces.append(z_raw)
        noise_all.append(noise)

    denoised_z_traces = np.array(denoised_z_traces)
    raw_z_traces = np.array(raw_z_traces)
    noise_all = np.array(noise_all)

    return denoised_z_traces, raw_z_traces, noise_all

print('loaded')


#### BE CAREFUL
resting_components = []
for i in os.listdir(fnames[0][:-5]+"_accepted"):
    resting_components.append(int(i[3:6]))

resting_components = np.sort(np.array(resting_components))
resting_components

#https://github.com/flatironinstitute/CaImAn/issues/802
indices_to_remove = np.delete(np.arange(0,len(cnm.estimates.idx_components),1), resting_components)
units_to_remove = cnm.estimates.idx_components[indices_to_remove]

num_components = cnm.estimates.C.shape[0]
new_inds = np.setdiff1d(np.arange(num_components), units_to_remove)

try:
    cnm.estimates.select_components(idx_components = new_inds, save_discarded_components=False)
except:
    print("check error...")

cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

from caiman.utils.visualization import get_contours

maxthr=0.2
nrgthr=0.9
thr_method = 'max'

thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
idx=cnm.estimates.idx_components
coordinates = get_contours(cnm.estimates.A[:,idx], np.shape(cn_filter), thr, thr_method)

good_inds = cnm.estimates.idx_components
denoised_traces = cnm.estimates.C[good_inds,:]
residuals = cnm.estimates.YrA[good_inds,:]
denoised_z_traces, raw_z_traces, noise_all = zscore_traces(denoised_traces,residuals, offset_method='floor')

import pickle
with open(fnames[0]+"_DF_filtered.txt","wb") as fp:
    pickle.dump(np.array(denoised_z_traces),fp)

plt.rcParams['figure.facecolor'] = 'white'
new_folder = fnames[0][:-5]+'_filtered_accepted'

import shutil

try:
    shutil.rmtree(new_folder)
except:
    print('Folder ok')

os.mkdir(new_folder)

for i in range(len(cnm.estimates.idx_components)):
#for i in [58]:
    f, axs = plt.subplots(2, 3, gridspec_kw={'height_ratios': [3, 1]}, figsize=(13,9))
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


    f.suptitle("Accepted Components ID {}, ID CaImAn: {}".format(i,cnm.estimates.idx_components[i]), fontsize=16)
    f.tight_layout()
    f.subplots_adjust(top=0.95)
    f.patch.set_facecolor('white')
    plt.savefig("{}/ID_{:03d}.png".format(new_folder,i), dpi=100)
    plt.close()
    print(i)
print("ok!")
import pickle
with open(fnames[0]+"_A_filtered.txt","wb") as fp:
    pickle.dump(cnm.estimates.A[:,idx],fp)


with open(fnames[0]+"_C_filtered.txt","wb") as fp:
    pickle.dump(cnm.estimates.C[good_inds,:],fp)

plt.rcParams['figure.facecolor'] = 'white'

from caiman.utils.visualization import get_contours

maxthr=0.2
nrgthr=0.9
thr_method = 'max'

thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
idx=cnm.estimates.idx_components
coordinates = get_contours(cnm.estimates.A[:,idx], np.shape(cn_filter), thr, thr_method)

plt.figure(figsize=(10,10))
plt.rcParams['figure.facecolor'] = 'white'
plt.imshow(cn_filter, cmap='viridis', interpolation='nearest')


for i in range(len(cnm.estimates.idx_components)):
    c = coordinates[i]
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    plt.plot(*v.T, c='white', lw=3)

    plt.plot(c["CoM"][1],c["CoM"][0], 'o', ms=10, color='black',markerfacecolor='None',markeredgewidth = 2.5)

plt.title("Accepted components: {}".format(len(cnm.estimates.idx_components)), fontsize = 16)
plt.tight_layout()
#plt.patch.set_facecolor('white')
plt.savefig('filtered_components_'+fnames[0][:-5]+'.png',dpi= 300)
plt.close()

import matplotlib.image as mpimg
img = mpimg.imread('components_'+fnames[0][:-5]+'.png')
img2 = mpimg.imread('filtered_components_'+fnames[0][:-5]+'.png')
fig, axs = plt.subplots(ncols=2, figsize=(10,5))

ax = axs[0]
ax.axis('off')
ax.imshow(img)

ax = axs[1]
ax.axis('off')
ax.imshow(img2)


plt.tight_layout()
plt.savefig('filtered_components_compare_'+fnames[0][:-5]+'.png', dpi=300)
plt.close()

def norm(a, rg=(0, 1)):
    a_norm = (a - a.min()) / (a.max() - a.min())
    return a_norm * (rg[1] - rg[0]) + rg[0]

import matplotlib as mpl
import matplotlib.cm as cmm

A = cnm.estimates.A[:,idx]

Ad = np.asarray(A.todense()).reshape((d1, d2, -1), order='F')
myfoot = []
for i in range(len(idx)):
    A_scl = norm(Ad[:, :, i], (0, 1))
#     smp = cmm.ScalarMappable(cmap='gray')
#     im_rgb = smp.to_rgba(cn_filter)[:, :, :3]
#     im_hsv = mpl.colors.rgb_to_hsv(im_rgb)
#     im_hsv_scl = im_hsv.copy()
#     im_hsv_scl[:, :, 2] = im_hsv[:, :, 2] * A_scl
    myfoot.append(A_scl)
myfoot = np.array(myfoot)

plt.rcParams['figure.facecolor'] = 'white'

footprints = myfoot
colormap='gist_rainbow'
composite_fov = np.zeros((footprints.shape[1], footprints.shape[2], 3))
cmap_vals = cmm.get_cmap(colormap)

plt.figure(figsize=(12,12))
np.random.seed(0)
for cell_id in range(footprints.shape[0]):
    # select a random color for this cell
    color = cmap_vals(np.random.rand())

    # assign the color to each of the three channels, normalized by the footprint peak
    for color_channel in range(3):
        composite_fov[:,:,color_channel] += color[color_channel]*footprints[cell_id]/np.max(footprints[cell_id])

# set all values > 1 (where cells overlap) to 1:
composite_fov[np.where(composite_fov > 1)] = 1


# annotate each cell with a label centered at its peak
for cell_id in range(footprints.shape[0]):
    peak_loc = np.where(footprints[cell_id]==np.max(footprints[cell_id]))
    plt.text(
        peak_loc[1][0],
        peak_loc[0][0],
        'cell {}'.format(str(cell_id).zfill(2)),
        color='white',
        ha='center',
        va='center',
        fontweight='bold',
    )

# show the image
plt.imshow(composite_fov)
plt.tight_layout()
plt.savefig('filtered_lights_'+fnames[0][:-5]+'.png',dpi= 300)
plt.close()

img = mpimg.imread('lights_'+fnames[0][:-5]+'.png')
img2 = mpimg.imread('filtered_lights_'+fnames[0][:-5]+'.png')

fig, axs = plt.subplots(ncols=2, figsize=(10,5))

ax = axs[0]
ax.axis('off')
ax.imshow(img)

ax = axs[1]
ax.axis('off')
ax.imshow(img2)


plt.tight_layout()
plt.savefig('filtered_lights_compare_'+fnames[0][:-5]+'.png', dpi=300)
plt.close()

cm.stop_server(dview=dview)

from sys import exit
exit()
