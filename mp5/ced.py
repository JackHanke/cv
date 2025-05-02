from PIL import Image
import numpy as np
import cv2 

# NOTE this code was written by Jack Hanke

# smooth image with gaussian filter
def gausssmoothing(img_in: Image, N: float = 3, Sigma: float = 3):
    im_array = np.asarray(img)
    x, y = im_array.shape

    # approx kernel found here: https://en.wikipedia.org/wiki/Kernel_(image_processing)
    approx_gaussian_kernel = np.array(
        [
            [1,2,1],
            [2,4,2],
            [1,2,1],
        ]
    )
    approx_gaussian_kernel = (1/16)*approx_gaussian_kernel

    return_arr = cv2.filter2D(src=im_array, ddepth=-1, kernel=approx_gaussian_kernel)
    return_im = Image.fromarray(return_arr.astype(np.uint8))

    return return_im

# NOTE debug to check if convolution code I wrote works
def debug_grad(img_in: Image):
    im_array = np.asarray(img)
    x, y = im_array.shape
    c_x = np.array(
        [
            [-1,0,1],
            [-2,0,2],
            [-1,0,1],
        ]
    )

    return_arr = cv2.filter2D(src=im_array, ddepth=-1, kernel=c_x)
    return_im = Image.fromarray(return_arr.astype(np.uint8))

    return return_im

# computes magnitude and theta of image gradient for img_in using sobel filters
def imagegradient(img_in: Image):
    im_array = np.asarray(img_in.convert('L'))
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
            c_y_val = sum([g_y[i+1][j+1]*im_array[anchor_i + i][anchor_j + j] for i in range(-1,2) for j in range(-1,2)])
            c_y_ed_array[anchor_i][anchor_j] = c_y_val
            
            c_x_val = sum([g_x[i+1][j+1]*im_array[anchor_i + i][anchor_j + j] for i in range(-1,2) for j in range(-1,2)])
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

    return magnitude, theta, c_y_ed_array, c_x_ed_array

# 
def findthreshold(magnitude):
    h = 0.8

    x, y = magnitude.shape
    max_val = round(np.max(magnitude))
    histogram = [0 for _ in range(max_val+1)]

    for i in range(x):
        for j in range(y):
            val = round(magnitude[i][j])
            histogram[val] += 1
    i = 0
    cumulative_sum = 0
    while i < len(histogram):
        cumulative_sum += histogram[i]
        if cumulative_sum > 0.8*x*y:
            t_high = i
            return 0.5*t_high, t_high
        i += 1    
    t_high = max_val
    return 0.5*t_high, t_high

#     
def nonmaximasupress(magnitudes, thetas, low, high):
    low_im_array = np.zeros(thetas.shape)
    high_im_array = np.zeros(thetas.shape)
    x, y = thetas.shape

    for i in range(1,x-1):
        for j in range(1,y-1):
            magnitude = magnitudes[i][j]
            theta = thetas[i][j]

            # if north south
            if (theta < np.pi/6 or theta > (11/6)*np.pi) or (theta > (5/6)*np.pi and theta < (7/6)*np.pi):
                if magnitudes[i][j+1] < magnitude and magnitudes[i][j-1] < magnitude:
                    if magnitude > high:
                        high_im_array[i][j] = 1
                    if magnitude > low:
                        low_im_array[i][j] = 1
            # if north east or 
            elif (theta < (1/3)*np.pi and theta > (1/6)*np.pi) or (theta > (4/3)*np.pi and theta < (7/6)*np.pi):
                if magnitudes[i-1][j+1] < magnitude and magnitudes[i+1][j-1] < magnitude:
                    if magnitude > high:
                        high_im_array[i][j] = 1
                    if magnitude > low:
                        low_im_array[i][j] = 1
            # if east west
            elif (theta < (2/3)*np.pi and theta > (1/3)*np.pi) or (theta > (5/3)*np.pi and theta < (4/6)*np.pi):
                if magnitudes[i-1][j] < magnitude and magnitudes[i+1][j] < magnitude:
                    if magnitude > high:
                        high_im_array[i][j] = 1
                    if magnitude > low:
                        low_im_array[i][j] = 1
            # if north west
            elif (theta < (5/6)*np.pi and theta > (2/3)*np.pi) or (theta > (11/6)*np.pi and theta < (5/3)*np.pi):
                if magnitudes[i+1][j-1] < magnitude and magnitudes[i-1][j+1] < magnitude:
                    if magnitude > high:
                        high_im_array[i][j] = 1
                    if magnitude > low:
                        low_im_array[i][j] = 1

    return low_im_array, high_im_array

# 
def edgelinking(img_in: Image, img_path: str):
    im_array = np.asarray(img)
    x, y = im_array.shape

    # initialize arrays
    return_array = np.zeros(im_array.shape)
    strong_edges = np.zeros(im_array.shape)
    weak_edges = np.zeros(im_array.shape)

    # smooth image
    smoothed_img_in = gausssmoothing(img_in=img_in)
    output_img_path = 'smoothed_' + img_path[:-3] + 'png'
    smoothed_img_in.save(output_img_path)

    # calculate magnitudes and thetas of image gradient
    magnitudes, thetas, im_array_y, im_array_x = imagegradient(img_in = smoothed_img_in)
    # first for c_y
    y_max = np.max(im_array_y)
    im_array_y = im_array_y * (255/y_max)
    output_img_path = 'y_' + img_path[:-3] + 'png'
    im_y = Image.fromarray(im_array_y.astype(np.uint8), 'L')
    im_y.save(output_img_path)
    # then for x_x
    x_max = np.max(im_array_x)
    im_array_x = im_array_x * (255/x_max)
    output_img_path = 'x_' + img_path[:-3] + 'png'
    im_x = Image.fromarray(im_array_x.astype(np.uint8), 'L')
    im_x.save(output_img_path)
    print(f'Max magnitudes: {np.max(magnitudes)}. Min magnitudes: {np.min(magnitudes)}')

    # do debug for self-check
    debug_x_im = debug_grad(img_in=img_in)
    output_img_path = 'debug_x_' + img_path[:-3] + 'png'
    im_x.save(output_img_path)

    # compute low and high image gradient thresholds
    low_threshold, high_threshold = findthreshold(magnitude=magnitudes)

    print(f'High threshold: {high_threshold}. Low threshold: {low_threshold}')

    low_im_array, high_im_array = nonmaximasupress(magnitudes=magnitudes, thetas=thetas, low=low_threshold, high=high_threshold)
    # save low im array
    output_img_path = 'low_' + img_path[:-3] + 'png'
    return_im = Image.fromarray((255*low_im_array).astype(np.uint8), 'L')
    return_im.save(output_img_path)

    # save high im array
    output_img_path = 'high_' + img_path[:-3] + 'png'
    return_im = Image.fromarray((255*high_im_array).astype(np.uint8), 'L')
    return_im.save(output_img_path)

    # begin edge linking
    return_array = np.zeros(im_array.shape)
    for i in range(x):
        for j in range(y):
            # link edges by strong array
            if high_im_array[i][j] == 1:
                edge_link(
                    low_array=low_im_array,
                    high_array=high_im_array,
                    return_array=return_array,
                    pixel_coords=(i,j)
                )
                
    # save final image with "canny_" prefix
    output_img_path = 'canny_' + img_path[:-3] + 'png'
    return_im = Image.fromarray((255*return_array).astype(np.uint8), 'L')
    return_im.save(output_img_path)
    return return_im

# helper function for recursion in edge linking
def edge_link(
        low_array: np.array,
        high_array: np.array,
        return_array: np.array,
        pixel_coords: tuple
    ):
    # get coords
    i, j = pixel_coords

    # if return array is 1 ie "already seen", return
    if return_array[i][j] == 1: return

    # if not either edge, return
    if low_array[i][j] == 0: return
    else:
        return_array[i][j] = 1

        # recurse for each neighbor
        neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
        for neighbor in neighbors:
            edge_link(
                low_array=low_array,
                high_array=high_array,
                return_array=return_array,
                pixel_coords=neighbor
            )

if __name__ == '__main__':
    file_names = [
        "lena.bmp",
        "pointer1.bmp",
        "test1.bmp",
        "joy1.bmp",
    ]

    for file_name in file_names:
        # grayscale image
        img = Image.open(file_name).convert('L')
        img.save(file_name[:-3]+'png') # for report
        edges = edgelinking(img_in = img, img_path=file_name)
