import numpy as np
from os import listdir
from descriptors import haralick_bit
from paths import kth_dir, kth_path
import cv2

def main():
    # List for all the signatures from images with BiTdec, etc..
    listOflists = list()
    print('Extracting features ....')
    # Loop in path and grab subfolders
    counter = 0
    for kth_class in kth_dir:
        print(f'Current folder: {kth_class}')
        # Grab files from subfolders
        for filename in listdir(kth_path + kth_class + '/'):
            counter += 1
            img_name = f'{kth_path}{kth_class}/{filename}'
            # Read/Load Image as gray
            img = cv2.imread(img_name, 0)
            features = np.concatenate((haralick_bit(img), np.array([kth_class]), np.array([img_name])), axis=None)
            print(f' Image count: {counter}')
            # Add image features to listOflists
            listOflists.append(features)
    final_array = np.array(listOflists)
    np.save('cbir_signaturesHAR_BIT.npy', final_array)
    print('Extraction concluded successfully!')

if __name__ == "__main__":
    main()
