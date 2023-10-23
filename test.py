from skimage import io
import numpy as np
from matplotlib.colors import SymLogNorm
from skimage.transform import rotate
from sklearn.preprocessing import normalize
from skimage import exposure
import matplotlib.pyplot as plt
from utils import align_single


base = '/Volumes/Zunisha/Registration/'
im = io.imread(base + 'aligned/TRAP_DL_4_561.tiff')
aligned = align_single(im)
io.imsave(base + 'test_aligned_1.tiff', aligned)
