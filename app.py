import streamlit as st
import cv2, os, time
import numpy as np
from PIL import Image
import descriptors
import pandas as pd
from scipy.spatial import distance
from distances import distance_selection
from upload import upload_file
from sklearn.preprocessing import StandardScaler


def main():

    # Display a title
    st.title("CBIR")
    st.write("App launched!")

  
    # Get user input for number of similar images to display
    input_value = st.sidebar.number_input("Enter a value", min_value=1, max_value=500, value=10, step=1)
    st.sidebar.write(f"You entered {input_value}")

    # Display a dropdown to choose distance calculation method
    options = ["Euclidean", "Canberra", "Manhattan", "Chebyshev", "Minkowsky"]
    distance_option = st.sidebar.selectbox("Select a distance", options)
    st.sidebar.write(f"You chose {distance_option}")

    # Get user input for image descriptor method
    descriptor_method = st.sidebar.selectbox("Select Image Descriptor Method",
                                             ["bitdesc", "glcm", "haralick", "bitdesc + haralick"])

    # Get user input for normalization 
    normalize = st.sidebar.radio("Normalize", ["Yes", "No"], index=1)

    
    
    distanceList = list()

    # Check if an image is uploaded
    is_image_uploaded = upload_file()
    if is_image_uploaded:
        # Display section title
        st.write('''
                 # Search Results
                 ''')

        # Path to the uploaded query image
        query_image = 'uploaded_images/query_image.png'
        img = cv2.imread(query_image, 0)
        

        if descriptor_method == "bitdesc":
            bit_feat = descriptors.bitdesc(img)
            signatures = np.load('cbir_signatures_v1.npy')
        elif descriptor_method == "glcm":
            bit_feat = descriptors.glcm(img)
            signatures = np.load('cbir_signaturesGLCM_v1.npy')
        elif descriptor_method == "haralick":
            bit_feat = descriptors.haralick_fct(img)
            signatures = np.load('cbir_signaturesHAR_v1.npy')
        elif descriptor_method == "bitdesc + haralick":
            bit_feat = descriptors.haralick_bit(img)
            signatures = np.load('cbir_signaturesHAR_BIT.npy')
        
        

        # Normalize uploaded image if selected
        if normalize == "Yes":
            scaler = StandardScaler()
            bit_feat = np.array(bit_feat).reshape(1, -1)
            bit_feat = scaler.fit_transform(bit_feat).flatten()

            normalized_signatures = []
            for sign in signatures:
                features = np.array(sign)[0:-2].astype('float')
                normalized_features = scaler.fit_transform(features.reshape(1, -1)).flatten()
                normalized_signatures.append(list(normalized_features) + sign[-2:].tolist())
            signatures = np.array(normalized_signatures)

        # Calculate distances between query image and signatures
        for sign in signatures:
            sign = np.array(sign)[0:-2].astype('float')
            sign = sign.tolist()
            calculated_distance = distance_selection(distance_option, bit_feat, sign)
            distanceList.append(calculated_distance)

        st.write("Distances computed successfully")

        # Find indices of the most similar images
        minDistances = np.argsort(distanceList)[:input_value]
        image_paths = [signatures[small][-1] for small in minDistances]
        classes = [signatures[small][-2] for small in minDistances]

        # Count unique classes and create a DataFrame for plotting
        unique_values, counts = np.unique(classes, return_counts=True)
        list_classes = list()
        print("Unique value with their counts")
        for value, count in zip(unique_values, counts):
            print(f"{value}:{count}")
            list_classes.append(value)

        df = pd.DataFrame({"Value": unique_values, "frequency":counts})
        st.bar_chart(df.set_index("Value"))

        # Display the most similar images
        st.write('# Similar Images')
        for i, (path, class_) in enumerate(zip(image_paths, classes)):
            st.write(f"## Similar Image {i + 1}")
            st.image(path, caption=f"Class: {class_}", use_column_width=True, width=100)
            ##st.write(f"Path {path}")

    else:
        st.write("Welcome! Please upload an image to get started ...")

if __name__ == "__main__":
    main()
