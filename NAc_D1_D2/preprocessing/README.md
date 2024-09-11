# Code Overview for Preprocessing Steps

**1. File: std.py**
+ **Purpose:** This script is responsible for the initial preprocessing, which includes:

**Motion Correction**: Aligning video frames to reduce movement artifacts.

**CNMFe:** Running the CNMF-e algorithm for source extraction. (For more details on using this package and appropriate documentation, please visit https://github.com/flatironinstitute/CaImAn.)

**DF/F Traces**: Calculating the ΔF/F traces, which represent the change in fluorescence relative to baseline.

+ **Important Note:** The input video for preprocessing must be in TIFF format and named MC.tiff.

+ **Key Variables and Their Default Values:**
merge_thr = 0.995: The merging threshold, which is the maximum allowed correlation when combining potential cell contours.

min_corr = 0.6: The minimum peak value from the correlation image, used to identify potential cells.

min_pnr = 3.5: The minimum peak-to-noise ratio from the PNR (Peak-to-Noise Ratio) image, which helps in detecting cells.

gSig = (5, 5): The width of the Gaussian kernel in 2D, approximating the size of a neuron.

gSiz = (10, 10): The estimated diameter of a neuron, generally set to 4 times the Gaussian width plus 1.


**2. File: selecting.py**
+ **Purpose:** After running the CNMFe algorithm, this script launches a graphical interface. It allows users to manually review and select the neurons that were automatically detected by CNMFe.

**3. File: after1.py**
+ **Purpose:** Following the manual selection of neurons, this script extracts and saves the ΔF/F traces for the cells that were classified as "good."
  
**4. File: after2.py**
+ **Purpose:** Similar to after1.py, but this script extracts and saves the ΔF/F traces for both "good" and "maybe" cells, providing a broader dataset.

  
