from PIL import Image
import numpy as np
import cv2

# NOTE this code was written by Jack Hanke


def line_detection(img_in: Image, file_name: str):
    im_array = np.asarray(img_in)
    y, x = im_array.shape # NOTE switched to align with axis labeling from slides

    # get edges from OpenCV Canny edge detection with low threshold 100 and high threshold 200
    edges = cv2.Canny(im_array, 100, 200)

    # quantize rho
    num_rhos = 200
    low_rho = -np.sqrt(x**2 + y**2)
    high_rho = np.sqrt(x**2 + y**2)
    rhos = np.linspace(low_rho, high_rho, num=num_rhos)
    # quantize theta
    num_thetas = 200
    low_theta = -np.pi/2
    high_theta = np.pi/2
    thetas = np.linspace(low_theta, high_theta, num=num_thetas)

    parameter_space = np.zeros((num_thetas, num_rhos))

    for i in range(y):
        for j in range(x):
            pixel = edges[i][j]
            # if strongest point of an edge map
            if pixel == 255:
                for theta_index, theta in enumerate(thetas):
                    raw_rho = j*np.cos(theta) + i*np.sin(theta)
                    # find index of closest rho
                    rho_index = 0
                    while raw_rho > rhos[rho_index] or rho_index == num_rhos:
                        rho_index += 1

                    # vote for parameters
                    parameter_space[theta_index][rho_index] += 1

    # renormalize for viewing
    max_vote = np.max(parameter_space)
    parameter_space = parameter_space * (255/max_vote)
    pspace_im = Image.fromarray(parameter_space.astype(np.uint8))
    pspace_im.save('parameter_space_'+file_name[:-3]+'png')

    # dilate parameter space with 1s filter
    kernel = np.ones((7,7))
    dilated_parameter_space = cv2.dilate(parameter_space, kernel)
    # dilated_parameter_space = cv2.dilate(dilated_parameter_space, kernel)
    pspace_im = Image.fromarray(dilated_parameter_space.astype(np.uint8))
    pspace_im.save('dilated_parameter_space_'+file_name[:-3]+'png')

    # find local maximum above a threshold percentage 
    sig_threshold = 0.5
    localmax_parameter_space = parameter_space*(parameter_space >= dilated_parameter_space)*(parameter_space > (255 * sig_threshold))
    # view parameter space
    pspace_im = Image.fromarray(localmax_parameter_space.astype(np.uint8))
    pspace_im.save('localmax_parameter_space_'+file_name[:-3]+'png')

    # get significant intersections from localmax_parameter_space array

    significant_parameters = []
    for theta_index in range(num_thetas):
        for rho_index in range(num_rhos):
            if localmax_parameter_space[theta_index][rho_index] > 0:
                sig_rho, sig_theta = rhos[rho_index], thetas[theta_index]
                significant_parameters.append((-sig_rho, -sig_theta)) # NOTE there is a conversion error somewhere, this is a hack to fix

    # superimpose found lines
    return_array = np.copy(im_array)
    for sig_rho, sig_theta in significant_parameters:
        b = sig_rho / np.sin(sig_theta)
        m = 1/np.tan(sig_theta)
        for i in range(x):
            pred_y = int(m*i + b)
            if pred_y >= 0 and pred_y <= y-1:
                return_array[pred_y][i] = 255

    return_im = Image.fromarray(return_array.astype(np.uint8))
    return_im.save('edgified_'+file_name[:-3]+'png')
    return return_im

if __name__ == '__main__':
    file_names = [
        'test.bmp',
        'test2.bmp',
        'input.bmp',
    ]

    for file_name in file_names:
        img_in = Image.open(file_name).convert('L')
        img_in.save(file_name[:-3]+'png') # for report
        img_out = line_detection(img_in = img_in, file_name = file_name)


