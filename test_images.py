import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from skimage import io
from tqdm import tqdm
from skimage.transform import rotate
from sklearn.preprocessing import normalize


# TODO


def intersection_over_union(im1, im2):
    # implements the intersection over union metric on two images
    img1 = normalize(im1)
    img2 = normalize(im2)
    intersection = np.multiply(img1, img2)
    union = np.add(img1, img2)
    return np.sum(intersection) / np.sum(union)


def find_best_angle(im):
    # finds the best angle for a given image
    iou_max = 0
    rot_angle = 0
    for i, a in enumerate(np.linspace(-45, 45, 50)):
        temp_rot_slice = rotate(im, angle=a, mode='constant', cval=np.min(im))
        iou = intersection_over_union(temp_rot_slice, temp_rot_slice[:, ::-1])
        if iou > iou_max:
            iou_max = iou
            rot_angle = a
    return rot_angle


def mean_stack_angle(im):
    angles = []
    print('Finding best angle for each slice in stack')
    midpoint = im.shape[0] // 2
    for i in tqdm(range(midpoint-100, midpoint+100, 10)):
        angles.append(find_best_angle(im[i, :, :]))
    return np.mean(angles)


def find_image_centroid(im):
    # finds the centroid of the image
    # find the x and y coordinates of the image
    x = np.linspace(0, im.shape[0] - 1, im.shape[0])
    y = np.linspace(0, im.shape[1] - 1, im.shape[1])
    # find the x and y coordinates of the centroid
    x_centroid = np.sum(np.multiply(im, np.reshape(x, [-1, 1]))) / np.sum(im, dtype=np.float64)
    y_centroid = np.sum(np.multiply(im, np.reshape(y, [1, -1]))) / np.sum(im, dtype=np.float64)
    return x_centroid, y_centroid


def shift_to_center(im):
    # shift the image so that the centroid is at the center
    x_centroid, y_centroid = find_image_centroid(im)
    x_shift = int(im.shape[0] / 2 - x_centroid)
    y_shift = int(im.shape[1] / 2 - y_centroid)
    shifted = np.roll(im, x_shift, axis=0)
    shifted = np.roll(shifted, y_shift, axis=1)
    return shifted


def shift_to_centroid(im, centroid):
    # shift the image so that the centroid is at the center
    x_shift = int(im.shape[0] / 2 - centroid[0])
    y_shift = int(im.shape[1] / 2 - centroid[1])
    shifted = np.roll(im, x_shift, axis=0)
    shifted = np.roll(shifted, y_shift, axis=1)
    return shifted


def smooth_centroids(centroids, window_size=20):
    # smooths the centroids using a moving average
    new_centroids = np.array(centroids)
    kernel = np.ones(window_size) / window_size
    new_centroids[:,0] = np.convolve(new_centroids[:,0], kernel, mode='same')
    new_centroids[:,1] = np.convolve(new_centroids[:,1], kernel, mode='same')
    return new_centroids


def gradient_magnitude(centroids):
    # calculates the gradient magnitude of the centroids
    dx = np.gradient(centroids[:,0])
    dy = np.gradient(centroids[:,1])
    return np.sqrt(np.add(np.square(dx), np.square(dy)))


if __name__ == '__main__':
    base = '/Volumes/Zunisha/Registration/'
    #list all filenames with extension .tiff
    filenames = 

    print('Loading image')
    im = io.imread('/Volumes/Zunisha/Registration/TRAP_DL_12_561.tiff')

    print('Shifting image to center')
    shifted_im = np.zeros_like(im, dtype=np.float16)
    centroids = []
    for i in tqdm(range(im.shape[0])):
        slice = im[i, :, :]
        slice = np.array((slice - np.mean(slice)) / np.std(slice), dtype=np.float16)
        #slice = slice - slice.min()
        slice[np.where(slice<=0)] = 0
        shifted_im[i, :, :] = slice
        centroids.append(find_image_centroid(slice))

    centroids = smooth_centroids(centroids)
    good_interval = np.where(gradient_magnitude(centroids) < 5)[0]
    low_cut, up_cut = good_interval.min(), good_interval.max()
    for i in range(low_cut, up_cut):
        shifted_im[i,:,:] = shift_to_centroid(shifted_im[i,:,:], centroids[i])

    rot_angle = mean_stack_angle(shifted_im)

    # Take a slice from the original and processed images and plot them
    processed_im = np.zeros_like(im, dtype=np.uint16)
    for idx in tqdm(range(low_cut, up_cut)):
        #slice = im[idx, :, :]
        # 0  # norm='symlog'
        # plt.show()

        shifted_slice = shifted_im[idx, :, :]
        rot_slice = rotate(shifted_slice, angle=rot_angle,
                           mode='constant', cval=np.min(shifted_slice))
        processed_im[idx, :, :] = np.array((2**16)*SymLogNorm(0.01, clip=True).__call__(rot_slice),
                                           dtype=np.uint16)
        # rot_slice = rotate(shifted_slice, angle=rot_angle, mode='constant', cval=np.min(shifted_slice))
        # plt.imshow(rot_slice, norm='symlog', cmap='gray')  # norm='symlog'
        # plt.show()

    print('Saving image')
    io.imsave('./Images/TRAP_DL_12_561_rough.tiff', processed_im)
