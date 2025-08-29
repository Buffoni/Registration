import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from skimage import io
from skimage.util import img_as_ubyte
import ants
from skimage.transform import rotate, resize
from sklearn.preprocessing import normalize
import pickle
from utils import *
import yaml
from scipy.spatial.transform import Rotation

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

moving_filename = config['moving_filename']
target_filename = config['target_filename']
contrast_min = config['contrast_min']
contrast_max = config['contrast_max']
output_path = config['output_path']

print("Loading moving image:", moving_filename)
im = io.imread(moving_filename)
print("Loading target image:", target_filename)
fixed = io.imread(target_filename)

print("Normalizing contrast...")
# contrast normalization
im_min = np.min(im)
im_max = np.max(im)

imr = (im - im_min) / (contrast_max - im_min)
imr[np.where(imr > 1)] = 1
imr[np.where(imr < contrast_min / 255)] = 0


print("Coarse registration...")
rot_angle = mean_stack_angle(imr)
if config['flip']:
    rot_angle = rot_angle + 180

for idx in range(imr.shape[0]):
    shifted_slice = imr[idx, :, :]
    rot_slice = rotate(shifted_slice, angle=rot_angle,mode='constant', cval=np.min(shifted_slice))
    imr[idx, :, :] = rot_slice

shifted_im = np.zeros_like(imr, dtype=np.float64)
for i in range(imr.shape[0]):
    slice = imr[i, :, :]
    shifted_im[i, :, :] = slice

if config['z_cut'] == "auto":
    centroids = []
    for i in range(shifted_im.shape[0]):
        slice = shifted_im[i, :, :]
        centroids.append(find_image_centroid(slice))

    centroids = smooth_centroids(centroids)
    thresh = np.mean(gradient_magnitude(centroids)[len(centroids)//2 - 50 : len(centroids)//2 + 50])
    good_interval = np.where(gradient_magnitude(centroids) < thresh*10)[0]
    low_cut, up_cut = good_interval.min(), good_interval.max()
else:
    low_cut, up_cut =  config['z_cut']

for i in range(shifted_im.shape[0]):
    if (i < low_cut) or (i > up_cut):
        shifted_im[i, :, :] = 0


if config['y_cut'] == "auto":
    centroids = []
    for i in range(shifted_im.shape[1]):
        slice = shifted_im[:, i, :]
        centroids.append(find_image_centroid(slice))

    centroids = smooth_centroids(centroids)
    thresh = np.mean(gradient_magnitude(centroids)[len(centroids)//2 - 50 : len(centroids)//2 + 50])
    good_interval = np.where(gradient_magnitude(centroids) < thresh*10)[0]
    low_cut, up_cut = good_interval.min(), good_interval.max()
else:
    low_cut, up_cut =  config['y_cut']

for i in range(shifted_im.shape[1]):
    if i < low_cut or i > up_cut:
        shifted_im[:, i, :] = 0

if config['x_cut'] == "auto":
    centroids = []
    for i in range(shifted_im.shape[2]):
        slice = shifted_im[:, :, i]
        centroids.append(find_image_centroid(slice))

    centroids = smooth_centroids(centroids)
    thresh = np.mean(gradient_magnitude(centroids)[len(centroids)//2 - 50 : len(centroids)//2 + 50])
    good_interval = np.where(gradient_magnitude(centroids) < thresh*10)[0]
    low_cut, up_cut = good_interval.min(), good_interval.max()
else:
    low_cut, up_cut =  config['x_cut']

for i in range(shifted_im.shape[2]):
    if i < low_cut or i > up_cut:
        shifted_im[:, :, i] = 0

fixed_im = ants.from_numpy(fixed, spacing=config["voxel_size_target"])
moving_im = ants.from_numpy(img_as_ubyte(shifted_im), spacing=config["voxel_size_moving"])
if config['match_shapes']==True:
    print("Rescaling image to target shape...")
    moving_im=ants.resample_image_to_target(moving_im,fixed_im)


print("Fine registration (ANTs)...")
mytx = ants.registration(fixed=fixed_im, moving=moving_im, type_of_transform='SyNAggro', reg_iterations=(40, 20, 10, 0),
                         outprefix=output_path)
#check if output path exists and create if not
if not os.path.exists(output_path):
    os.makedirs(output_path)

rot = Rotation.from_euler('z', rot_angle, degrees=True)
affine_matrix = np.eye(4)
affine_matrix[:3, :3] = rot.as_matrix()

# Create ANTsPy affine transform
affine_transform = ants.create_ants_transform(transform_type='AffineTransform', matrix=affine_matrix)
ants.write_transform(affine_transform, output_path + 'initial_affine.mat')

trans = [output_path + 'initial_affine.mat']
for t in mytx['fwdtransforms']:
    trans.append(t)

with open(output_path + 'transforms_order.pkl','wb') as f:
    pickle.dump(trans,f)

warpedimage = ants.apply_transforms( fixed=fixed_im, moving=moving_im, transformlist=mytx['fwdtransforms'])

io.imsave(output_path + 'reg_image.tiff', warpedimage.numpy())
print('Saved image and transformation to', output_path)

