from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

# NOTE this code was written by Jack Hanke

# sum of squared distances
def ssd(candidate: np.array, template: np.array):
    return np.sum(np.square(candidate - template))

# cross correlation
def cc(candidate: np.array, template: np.array):
    return np.sum(np.tensordot(candidate, template, axes=(0,1)))

# normalized cross correlation
def ncc(candidate: np.array, template: np.array):
    candidate_bar = np.mean(candidate, axis=(0,1))
    template_bar = np.mean(template, axis=(0,1))

    candidate_hat = candidate - candidate_bar
    template_hat = template - template_bar

    a = np.sum(np.square(template_hat))
    b = np.sum(np.square(candidate_hat))

    return cc(template=template_hat, candidate=candidate_hat) / np.sqrt(a*b)

# get corners of square from tuple
def get_square_from_target(target: tuple, size: int, x:int, y:int):
    i, j = target

    # bounds
    i_1 = i-size
    if i_1 < 0: i_1 = 0
    i_2 = i+size 
    if i_2 > x: i_2 = x

    j_1 = j-size
    if j_1 < 0: j_1 = 0
    j_2 = j+size
    if j_2 > x: j_2 = y

    return [(i_1,j_1), (i_2,j_2)]

# draw box on image
def draw_box(img_in: Image, target: tuple, size: int = 20):
    im_array = np.asarray(img_in)
    try:
        x, y, c = im_array.shape
    except ValueError:
        x, y = im_array.shape

    corners = get_square_from_target(target=target, size=size, x=x, y=y)
    
    draw = ImageDraw.Draw(img_in)
    draw.rectangle(corners, outline='red', width=2)
    return img_in

# takes previous region and target, finds highest match by given criterion within search window
def object_find(im: Image, previous_region: np.array, previous_target: tuple, matcher_name: str, matcher: callable, size: int):
    im_array = np.asarray(im)
    try:
        x, y, c = im_array.shape
    except ValueError:
        x, y = im_array.shape

    prev_i, prev_j = previous_target
    
    # 
    region, target = None, None
    best_score = -float('inf')

    # define a search window
    window_size = 10
    for i in range(-window_size,window_size+1):
        for j in range(-window_size,window_size+1):
            try_i = prev_i + i
            try_j = prev_j + j
            try_target = (try_i, try_j)

            corners = get_square_from_target(target=try_target, size=size, x=x, y=y)

            (i1,j1), (i2,j2) = corners
            try_region = im_array[j1:j2+1, i1:i2+1,]

            # prevents out of bounds box checks
            if try_region.shape == (2*size+1, 2*size+1, 3) or try_region.shape == (2*size+1, 2*size+1):
                
                # compute match score for trial region
                score = matcher(candidate=try_region, template=previous_region)

                # regularization
                # score *= (1/(((i + window_size+1)**2)*((j+window_size+1)**2)))

                if score > best_score:
                    best_score = score
                    region = try_region
                    target = try_target

    return region, target

# image tracking over image_names array
def motion_track(image_names:list[Image], region, target: tuple, matcher_name: str, matcher:callable, size: int):
    images = []
    for image_name in image_names:
        im = Image.open(image_name)
        gray_im = im.convert('L')
        # calculate target
        region, target = object_find(
            im=gray_im,
            previous_region=region,
            previous_target=target,
            matcher_name=matcher_name,
            matcher=matcher,
            size=size,
        )
        # superimpose boxed region on image
        im_out = draw_box(
            img_in=im,
            target=target,
            size=size,
        )
        images.append(im_out)

    # make gif of final movie
    images[0].save(
        f'{matcher_name}.gif',
        save_all=True, 
        append_images=images[1:], 
        optimize=False, 
        duration=50, 
        loop=0,
    )


if __name__ == '__main__':
    # make initial bounding box image
    img_in = Image.open('image_girl/0001.jpg')
    gray_img_in = img_in.convert('L')
    im_array = np.asarray(gray_img_in)
    try:
        x, y, c = im_array.shape
    except ValueError:
        x, y = im_array.shape

    # initial parameters found by trial and error. size stays constant throughout
    start_target = (70, 45)
    size = 18

    corners = get_square_from_target(target=start_target, size=size, x=x, y=y)
    (i1,j1), (i2,j2) = corners
    start_region = im_array[j1:j2+1, i1:i2+1,]

    img_out = draw_box(img_in=img_in, target=start_target, size=size)
    img_out.save(f'image_girl/0000.jpg')

    # make image list
    image_names = []
    for filename in os.listdir('image_girl'):
        image_names.append('image_girl/' + filename)
    image_names.sort()

    # methods dictionary
    methods_dict = {
        'ssd': ssd,
        'cc': cc,
        'ncc': ncc,
    }

    # motion track over the course of 500 ims for each matching method
    for key, value in methods_dict.items():
        motion_track(
            image_names=image_names[1:], 
            region=start_region,
            target=start_target, 
            matcher_name=key, 
            matcher=value,
            size=size,
        )
        print(f'Generated {key}.gif')
