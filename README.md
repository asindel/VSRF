# Volumetric Super-Resolution Forests (VSRF)
Volumetric Super-Resolution Forests for MRI Super-Resolution

This is the source code to the conference article accepted at ICIP 2018.

## Citation
The source code is only for academic use, no commercial use is allowed.
If you use or adapt our code in your research, please cite our [ICIP article](https://ieeexplore.ieee.org/document/8451320):

	  @INPROCEEDINGS{8451320,
		author={A. Sindel and K. Breininger and J. K\"a{\ss}er and A. Hess and A. Maier and T. K\"ohler},
		booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
		title={Learning from a Handful Volumes: MRI Resolution Enhancement with Volumetric Super-Resolution Forests},
		year={2018},
		pages={1453-1457},
		doi={10.1109/ICIP.2018.8451320},
		ISSN={2381-8549},
		month={Oct}
	  }

## Requirements
- The code was developed with MATLAB R2017a and tested under Windows 10 (for some functions, e.g. imresize3 MATLAB 2017a is required)
- Required libraries/toolboxes:
	- [libEigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (for the mex/cpp files)
	- [SPM12](http://www.fil.ion.ucl.ac.uk/spm/software/spm12/) (to convert .nii to .mat and save the super-resolved images as .nii)
- Files from MATLAB file exchange:
	- [uipickfiles.m](https://de.mathworks.com/matlabcentral/fileexchange/10867-uipickfiles--uigetfile-on-steroids)
	- optional: [imshow3Dfull.m](https://de.mathworks.com/matlabcentral/fileexchange/47463-imshow3dfull--3d-imshow-in-3-views-)

## Usage
Run the example script to get an impression for the usage of VSRF. Some settings have to be specified as described in the code.

### Data
To run the example code with the Kirby 21 human brain MRI [1] dataset you can download the MPRAGE files [here](https://www.nitrc.org/frs/?group_id=313).
A script for preprocessing the data and converting the .nii files to .mat file is provided in `\preprocessing`.

### Building MEX Files
A pre-compiled MEX file is provided for Windows 64 bit, for other operating systems the MEX file can be build with the script `method\compile_forestRegrInference.m`.

### Training
VSRF is fast to train, depending on your hardware you can run the training in parallel.
For training either a set of low- and high-resolution MR volumes (mat files) can be used or the high-resolution volumes can be downscaled. These settings have to be defined in the example script `run_VSRF.m`.
The training process stores a .mat file containing the learned tree structure in `\method\models` for further usage.

### Inference
After training the trained forest is applied to the defined test MR volumes (see `run_VSRF.m` and `main_VSRF_MRI.m`). Then the super-resolved results are evaluated with the peak signal-to-noise ratio (PSNR) and the structural similarity (SSIM) [2] and the volumes are written to disk.

The code is based on [3], [4] and [5].

@author Aline Sindel

## References
[1] B. A. Landman, A. J. Huang, A. Gifford, D. S. Vikram, I. A. L. Lim, J. A. D. Farrell, J. A. Bogovic, J. Hua, M. Chen, S. Jarso, S. A. Smith, S. Joel, S. Mori, J. J. Pekar, P. B. Barker, J. L. Prince, and P. C. M. van Zijl, “Multi-parametric neuroimaging reproducibility: A 3-T resource study,” NeuroImage, vol. 54, no. 4, pp. 2854 – 2866, 2011.

[2] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.

[3] S. Schulter, C. Leistner, and H. Bischof. Fast and accurate image upscaling with super-resolution forests. CVPR 2015.

[4] R. Timofte, V. De Smet, L. van Gool. Anchored Neighborhood Regression for Fast Example-Based Super- Resolution. ICCV 2013."

[5] P. Dollar. Piotr's Computer Vision Matlab Toolbox (PMT).  http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
