import ants
import numpy as np
import matplotlib.pyplot as plt
import sys
base = '/Volumes/Zunisha/Registration/'

image_fixed= ants.image_read(base + 'aligned/TRAP_DL_12_561.tiff')
image_moving= ants.image_read(base + 'aligned/TRAP_DL_3_561.tiff')

print(image_fixed.shape)

moving = ants.resample_image(image_moving, np.asarray(image_fixed.shape)// 10, 1, 0)
fixed = ants.resample_image(image_fixed, np.asarray(image_fixed.shape)// 10, 1, 0)

mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyNRA', reg_iterations=(40, 20, 10, 0))
warpedimage = ants.apply_transforms( fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])

fixed = ants.resample_image(image_fixed, np.asarray(image_fixed.shape), 1, 0)
moving = ants.resample_image(warpedimage, np.asarray(image_fixed.shape), 1, 0)

mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyNRA', reg_iterations=(40, 20, 10, 0))
warpedimage = ants.apply_transforms( fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])

warpedimage.to_file(base + 'registered/TRAP_DL_3_561.tiff')

