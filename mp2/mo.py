from PIL import Image
import numpy as np

# erosion operation on img_in
def Erosion(img_in: str, im_name: str, kernel_size:int = 3, save: bool = True):
    # get array data and prepare final image
    im_array = np.asarray(img_in)
    # renormalize
    max_val = np.max(im_array)
    im_array = im_array/max_val
    
    return_im_array = np.zeros(im_array.shape)
    k = kernel_size // 2
    # get dims
    x_dim, y_dim = im_array.shape

    for anchor_i in range(k,x_dim-k):
        for anchor_j in range(k,y_dim-k):
            # is intersection all ones?
            positions = [im_array[i][j] for i in range(anchor_i-k, anchor_i+1+k) for j in range(anchor_j-k, anchor_j+1+k)]
            # if 0 is not in positions, then all 1s
            if 0 not in positions:
                return_im_array[anchor_i][anchor_j] = 255

    # format array as Image object
    return_im = Image.fromarray(return_im_array.astype(np.uint8))
    if save:
        # save image at im_path
        im_path = f'erosion_{im_name}.bmp'
        return_im.save(im_path)
        print(f'Image saved at: {im_path}')
    return return_im

# dilation operation on img_in
def Dilation(img_in: str, im_name: str, kernel_size:int = 3, save: bool = True):
    # get array data and prepare final image
    im_array = np.asarray(img_in)
    return_im_array = np.zeros(im_array.shape)
    # renormalize
    max_val = np.max(im_array)
    im_array = im_array/max_val

    k = kernel_size // 2
    # get dims
    x_dim, y_dim = im_array.shape

    for anchor_i in range(k,x_dim-k):
        for anchor_j in range(k,y_dim-k):
            # is intersection all ones?
            positions = [im_array[i][j] for i in range(anchor_i-k, anchor_i+1+k) for j in range(anchor_j-k, anchor_j+1+k)]
            # if 0 is not in positions, then all 1s
            if 1 in positions:
                return_im_array[anchor_i][anchor_j] = 255

    # format array as Image object
    return_im = Image.fromarray(return_im_array.astype(np.uint8))
    if save:
        # save image at im_path
        im_path = f'dilation_{im_name}.bmp'
        return_im.save(im_path)
        print(f'Image saved at: {im_path}')
    return return_im

# opening operation on img_in
def Opening(img_in: str, im_name: str, kernel_size:int = 3, save: bool = True):

    intermediate_im = Erosion(
        img_in = img_in,
        im_name = '',
        kernel_size = kernel_size,
        save = False
    )

    return_im = Dilation(
        img_in = intermediate_im,
        im_name = '',
        kernel_size = kernel_size,
        save=False
    )

    if save:
        # save image at im_path
        im_path = f'opening_{im_name}.bmp'
        return_im.save(im_path)
        print(f'Image saved at: {im_path}')

    return return_im
    
# closing operation on img_in
def Closing(img_in: str, im_name: str, kernel_size:int = 3, save: bool = True):

    intermediate_im = Dilation(
        img_in = img_in,
        im_name = '',
        kernel_size = kernel_size,
        save = False
    )

    return_im = Erosion(
        img_in = intermediate_im,
        im_name = '',
        kernel_size = kernel_size,
        save = False
    )

    if save:
        # save image at im_path
        im_path = f'closing_{im_name}.bmp'
        return_im.save(im_path)
        print(f'Image saved at: {im_path}')

    return return_im

def Boundary(img_in: str, im_name: str, kernel_size:int = 3, save: bool = True):
    # get array data and prepare final image
    im_array = np.asarray(img_in)
    # renormalize
    max_val = np.max(im_array)
    im_array = im_array/max_val

    return_im_array = np.zeros(im_array.shape)
    k = kernel_size // 2
    # get dims
    x_dim, y_dim = im_array.shape

    # get erosion of image
    eroded_im = Erosion(
        img_in = img_in,
        im_name = '',
        kernel_size=kernel_size,
        save= False
    )

    eroded_im_array = np.asarray(eroded_im)

    # subtract image from eroded 
    for i in range(x_dim):
        for j in range(y_dim):
            return_im_array[i][j] = 255*im_array[i][j] - eroded_im_array[i][j]

    # format array as Image object
    return_im = Image.fromarray(return_im_array.astype(np.uint8))
    if save:
        # save image at im_path
        im_path = f'boundary_{im_name}.bmp'
        return_im.save(im_path)
        print(f'Image saved at: {im_path}')
    return return_im

if __name__ == '__main__':
    im_list = [
        'gun',
        'palm',
    ]

    # run all function
    kernel_size = 3
    for im_name in im_list:
        im = Image.open(im_name+'.bmp')

        img_out = Erosion(img_in=im, im_name=im_name, kernel_size=kernel_size)
        img_out = Dilation(img_in=im, im_name=im_name, kernel_size=kernel_size)
        img_out = Opening(img_in=im, im_name=im_name, kernel_size=kernel_size)
        img_out = Closing(img_in=im, im_name=im_name, kernel_size=kernel_size)
        img_out = Boundary(img_in=im, im_name=im_name, kernel_size=kernel_size)

    # get good boundary
    
    for im_name in im_list:
        im = Image.open(im_name+'.bmp')

        intermediate = Closing(img_in=im, im_name=im_name, kernel_size=9, save=False)
        intermediate = Opening(img_in=intermediate, im_name=im_name, kernel_size=5, save=False)
        img_out = Boundary(img_in=intermediate, im_name='good_'+im_name, kernel_size=kernel_size)
        im_path = f'boundary_good_{im_name}.png'
        img_out.save(im_path)