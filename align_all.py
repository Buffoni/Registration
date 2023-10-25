import os
from utils import align_single
from skimage import io

#base = '/home/fis00lorebuff/Registration/Images/'
base = '/Volumes/Zunisha/Registration/'
filenames = [f for f in os.listdir(base) if f.endswith('.tiff') and not(f.startswith('.'))]
for f in filenames:
    print('Loading image' + f)
    im = io.imread(base + f)
    print('Align and rotate image')
    shifted_im = align_single(im)
    print('Saving image')
    io.imsave(base + 'aligned_new/' + f, shifted_im)