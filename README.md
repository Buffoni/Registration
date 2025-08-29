# 3D Image Registration Pipeline

This project performs coarse and fine registration of 3D microscopy images using image processing and ANTsPy.

## Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   
## Usage

Before running the pipeline, ensure you have correctly edited the configuration file `config.yaml` with appropriate paths and parameters.
- moving_filename: 'path/to/moving_image.tiff'
- target_filename: 'path/to/fixed_image.tiff'
- output_dir: 'where/to/save/results'
- contrast_min: 8-bit intensity below which all pixel values are set to 0 (remove background noise)
- contrast_max: original intensity above which all pixel values are set to 1 (remove saturation)
- z_cut: [min, max] z-slice range outside of which all pixel values are set to 0 (removes edge artifacts). If set to 'auto' tries to find automatically optimal bounding box
- x_cut and y_cut: same as z_cut but for x and y dimensions
- voxel_size: [x, y, z] voxel size in millimeters
- flip: bool whether the moving image is flipped with respect to the target image
- match_shapes: bool whether the moving image is resampled to match the shape of the target image

After this setup, run the pipeline:

 ```bash
   python3 main.py
   ```
## Output
The results will be saved in the specified output directory, including:
 - A pickle file containing the order of the transformations.
 - Individual files for each transformation step.
 - A TIFF file of the registered moving image.
