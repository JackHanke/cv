
from PIL import Image
import numpy as np

# NOTE this code is by Jack Hanke

# create skin color histogram from data in data directory
def make_histogram():
    # make hue, saturation histogram
    histogram = np.zeros((256,256))
    # loop over all data points
    dataset_size = 6
    for data_index in range(dataset_size):
        # convert to hue, saturation, value format
        img = Image.open(f'data/data{data_index}.png').convert('HSV')
        im_array = np.asarray(img)
        x, y, c = im_array.shape
        # loop over all pixels, increment if this pixel is seen
        for i in range(x):
            for j in range(y):
                pixel_hue, pixel_saturation = im_array[i][j][:2]
                histogram[pixel_hue][pixel_saturation] += 1

    # normalize histogram by maximum value seen
    max_value = np.max(histogram)
    historgram = histogram / max_value

    return histogram

# given a color image and a hue, saturation skin color histogram, produce a segmented image
def segment_with_histogram(img_path: str, histogram: np.array, threshold: float = 0.5):
    img_in = Image.open(img_path).convert('HSV')
    im_array = np.asarray(img_in)
    x, y, c = im_array.shape
    segmented_im = np.zeros((x,y))
    # loop over every pixel
    for i in range(x):  
        for j in range(y):
            # fetch pixel information
            pixel_hue, pixel_saturation, _ = im_array[i][j]
            # get probability from histogram that this pixel is a skin tone
            prob = histogram[pixel_hue][pixel_saturation]
            # if this probability is above given threshold, then it is a pixel of skin color
            if prob > threshold:
                segmented_im[i][j] = 255
    
    # create image and save to disk 
    output_img_path = 'segmented' + img_path[:-3] + 'png'
    return_im = Image.fromarray(segmented_im.astype(np.uint8))
    return_im.save(output_img_path)
    print(f'Image saved at: {output_img_path}')

    return return_im


if __name__ == '__main__':
    # make histogram from data in data directory
    histogram = make_histogram()
    
    file_names = [
        'gun1.bmp',
        'joy1.bmp',
        'pointer1.bmp',
    ]
    # for all files provided, create segmented image
    for file_name in file_names:
        segmented_im = segment_with_histogram(img_path=file_name, histogram=histogram, threshold=0.75)
        # convert bmp images for report
        img = Image.open(file_name)
        img.save(file_name[:-3]+'png')






