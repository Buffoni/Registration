import ants
import numpy as np
import matplotlib.pyplot as plt
import sys

image_fixed= ants.image_read('TRAP_DL_12_561.tiff')
image_moving= ants.image_read('TRAP_DL_3_561.tiff')

print(image_fixed.shape)

moving = ants.resample_image(image_moving, np.asarray(image_fixed.shape)// 3, 1, 0)#cosi' reshape stessa dimensione 
fixed = ants.resample_image(image_fixed, np.asarray(image_fixed.shape)// 3, 1, 0)

#(1093, 1383, 683) #attenzione questa e' solo una fig non tutte le fig hanno questa dimensione

mytx = ants.registration(fixed=fixed , moving=moving ,type_of_transform = 'SyN' )

mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])

mywarpedimage.to_file('test.tiff')

