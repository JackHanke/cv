from PIL import Image
import numpy as np

### NOTE Code written by Jack Hanke

# connect component labeling algorithm
def ccl(im_path: str):
    # open image and get dimensions
    im = Image.open(im_path+'.bmp')

    # get image data as np array
    im_array = np.asarray(im)
    x_dim, y_dim = im_array.shape

    # initialize initial and final images
    initial_labeled_im = np.zeros(im_array.shape)
    final_labeled_im = np.zeros(im_array.shape)

    # equivalence table class 
    class EquivalenceTable:
        def __init__(self):
            self.table = []

        # remove set with specific value for update
        def _get_set_relation(self, val):
            idx = 0
            while val not in self.table[idx]:
                idx += 1
                if idx == len(self.table): {val}
            return self.table.pop(idx)

        # update table with new value
        def add_new_relation(self, val):
            self.table.append({val})

        # combine the two sets that contain val1 and val2
        def update_relation(self, val1, val2):
            set1 = self._get_set_relation(val1)
            # if equivalent, put set1 back into table
            if val2 in set1:
                self.table.append(set1)
                return
            set2 = self._get_set_relation(val2)
            self.table.append(set1 | set2) # union the two sets
            return 

        # get smallest equivalent value
        def get_smallest_equivalent_index(self, val):
            for idx, relation in enumerate(self.table):
                if val in relation:
                    return idx

    # initialize equivalence table
    equiv_table = EquivalenceTable()

    # component_index
    component_index = 1
    for i in range(x_dim):
        for j in range(y_dim):
            # if no component, just continue
            if im_array[i][j] == 0:
                continue
            # else check
            else:
                # get neighboring pixel data, artifically pad image
                if i == 0 and j == 0: 
                    up_pixel_val = 0
                    left_pixel_val = 0
                elif i == 0:
                    up_pixel_val = 0
                    left_pixel_val = initial_labeled_im[i][j-1]
                elif j == 0:
                    up_pixel_val = initial_labeled_im[i-1][j]
                    left_pixel_val = 0
                else:
                    up_pixel_val = initial_labeled_im[i-1][j]
                    left_pixel_val = initial_labeled_im[i][j-1]
                
                # check and record equivalence relations as sets
                if up_pixel_val == 0 and left_pixel_val == 0:
                    initial_labeled_im[i][j] = component_index
                    # add to equivalence table
                    equiv_table.add_new_relation(val=component_index)
                    component_index += 1
                elif up_pixel_val == 0 and left_pixel_val != 0:
                    initial_labeled_im[i][j] = left_pixel_val
                elif up_pixel_val != 0 and left_pixel_val == 0:
                    initial_labeled_im[i][j] = up_pixel_val
                elif up_pixel_val != 0 and left_pixel_val != 0:
                    initial_labeled_im[i][j] = left_pixel_val
                    if up_pixel_val != left_pixel_val:
                        # update equivalence table
                        equiv_table.update_relation(val1=up_pixel_val, val2=left_pixel_val)

    # loop over initial labeled image 
    for i in range(x_dim):
        for j in range(y_dim):
            val = initial_labeled_im[i][j]
            if val == 0:
                final_labeled_im[i][j] = 0
            else:
                # get true label from equivalence table
                true_label = equiv_table.get_smallest_equivalent_index(val=val)
                
                # write to final iamge
                pixel = round(255*((true_label+1)/(len(equiv_table.table))))
                final_labeled_im[i][j] = pixel

    # return name for new image
    new_image_path = 'labeled_' + im_path + '.png'

    final = np.stack((final_labeled_im, np.zeros(final_labeled_im.shape), np.zeros(final_labeled_im.shape)), axis=2)

    final_im = Image.fromarray(final.astype(np.uint8))
    final_im.save(new_image_path)
    print(f'Image saved at: {new_image_path}')

    return final_im


if __name__ == '__main__':
    filenames = [
        'test',
        'face',
        'face_old',
        'gun',
    ]

    for filename in filenames:
        labled_im = ccl(im_path=filename)
