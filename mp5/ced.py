from PIL import Image
import numpy as np

import cv2 

# NOTE this code was written by Jack Hanke

#
def gausssmoothing(img_in: Image, N: float = 3, Sigma: float = 3):
    im_array = np.asarray(img)
    x, y = im_array.shape

    # temp = np.

    gauss_kernel = cv2.GaussianBlur(src=temp, ksize=3, Sigma=Sigma)

    

    print(gauss_kernel)



    return 

# computes magnitude and theta of image gradient for img_in using sobel filters
def imagegradient(img_in: Image):
    x, y = im_array.shape

    # define sobel filters
    g_y = np.array(
        [
            [1,2,1],
            [0,0,0],
            [-1,-2,-1],
        ]
    )
    g_x = np.array(
        [
            [-1,0,1],
            [-2,0,2],
            [-1,0,1],
        ]
    )

    c_y_ed_array = np.zeros(im_array.shape)
    c_x_ed_array = np.zeros(im_array.shape)
    # calculate partials
    for anchor_i in range(1,x-1):
        for anchor_j in range(1,y-1):
            # is intersection all ones?
            c_y_val = sum([c_y[i+1][j+1]*im_array[anchor_i + i][anchor_j + j]] for i in range(-1,2) for j in range(-1,2))
            c_y_ed_array[anchor_i][anchor_j] = c_y_val
            
            c_x_val = sum([c_x[i+1][j+1]*im_array[anchor_i + i][anchor_j + j]] for i in range(-1,2) for j in range(-1,2))
            c_x_ed_array[anchor_i][anchor_j] = c_x_val

    magnitude = np.zeros(im_array.shape)
    theta = np.zeros(im_array.shape)
    # calculate magnitude and angle from partials
    for i in range(x):
        for j in range(y):
            c_y_val = c_y_ed_array[i][j]
            c_x_val = c_x_ed_array[i][j]
            magnitude[i][j] = np.sqrt(c_y_val**2 + c_x_val**2)
            # NOTE in radians
            if c_x_val == 0:
                if c_y_val == 0:
                    theta[i][j] = 1
                else:
                    theta[i][j] = 1
            else:
                theta[i][j] = np.arctan(c_y_val/c_x_val)

    return magnitude, theta

def findthreshold():
    pass

def nonmaximasupress():
    pass

def edgelinking():
    pass

if __name__ == '__main__':
    file_names = [
        "lena.bmp",
        # "pointer1.bmp",
        # "test1.bmp",
        # "joy1.bmp",
    ]

    for file_name in file_names:
        # grayscale image
        img = Image.open(file_name).convert('L')

        smooth_im = gausssmoothing(img_in=img)
        smooth_im.show()

        



