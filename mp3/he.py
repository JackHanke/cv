from PIL import Image
import numpy as np

# NOTE this code is by Jack Hanke

# perform histogram equalization on an image
# if reg is None, perform no lighting correction
# if reg is linear, perform linear regression for lighting correction
#   y_hat = a_0 + a_1 i + a_2 j
# if reg is quadratic, perform quadratic regression for lighting correction
#   y_hat = a_0 + a_1 i + a_2 j + 
def HistoEqualization(img_in: Image, reg: str):
    # get array data and prepare final image
    im_array = np.asarray(img_in)
    return_im_array = np.zeros(im_array.shape)
    im_path = 'equalized_moon.png'
    # get dimensions
    x_dim, y_dim = im_array.shape

    # calculate histogram
    raw_hist = [0 for _ in range(256)]
    for i in range(x_dim):
        for j in range(y_dim):
            pixel = im_array[i][j]
            raw_hist[pixel] += 1

    # calculate cumulative histogram (cdf) and normalize
    cumulative_sum = 0
    cumulative_hist = []
    for pixel in raw_hist:
        cumulative_sum += pixel
        cumulative_hist.append((255*cumulative_sum)//(x_dim*y_dim))

    # add equalized pixel to return image
    for i in range(x_dim):
        for j in range(y_dim):
            original_pixel = im_array[i][j]
            equalized_pixel = cumulative_hist[original_pixel]
            return_im_array[i][j] = equalized_pixel

    # NOTE extra credit calculate lighting trend
    if reg is not None:
        # initialize data matrix
        X = []
        # initialize labels matrix
        b = []
        for i in range(x_dim):
            for j in range(y_dim):
                pixel = im_array[i][j]
                # make row of data matrix
                if reg == 'linear':
                    X.append([1, i, j])
                elif reg == 'quadratic':
                    X.append([1, i, j, i**2, i*j, j**2])
                # make row of labels matrix
                b.append([pixel])
        
        X = np.array(X)
        b = np.array(b)
        # compute intermediate values for pseudoinverse
        X_t = np.linalg.matrix_transpose(X)
        prod_inv = np.linalg.inv(X_t @ X)

        # directly compute optimal coefficients
        coefficients = prod_inv @ X_t @ b

        # subtract trend from each pixel
        trend_array = np.zeros(im_array.shape)
        for i in range(x_dim):
            for j in range(y_dim):
                equalized_pixel = return_im_array[i][j]
                if reg == 'linear':
                    data_vec = np.array([1, i, j])
                elif reg == 'quadratic':
                    data_vec = np.array([1, i, j, i**2, i*j, j**2])
                trend = data_vec @ coefficients
                trend_array[i][j] = trend
                return_im_array[i][j] = equalized_pixel - 0.25*(equalized_pixel - int(trend[0]))

        # trend image for debugging
        trend_im = Image.fromarray(trend_array.astype(np.uint8))
        trend_im.save('trend_' + reg + '_' + im_path)

    # format array as Image object
    return_im = Image.fromarray(return_im_array.astype(np.uint8))
    
    # 
    if reg is not None:
        im_path = reg + '_' + im_path

    # save image
    return_im.save(im_path)
    print(f'Image saved at: {im_path}')

    return return_im



if __name__ == '__main__':

    im = Image.open('moon.bmp').convert('L')
    im.save('moon.png')

    for reg in [None, 'linear', 'quadratic']:

        img_out = HistoEqualization(img_in=im, reg=reg)

