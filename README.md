# Content-Based Image Retrieval (CBIR) with Streamlit
This project implements a Content-Based Image Retrieval (CBIR) system using Streamlit, OpenCV, and scikit-learn. CBIR allows users to search for similar images in a dataset based on the content of a query image. The application supports various distance metrics and image descriptor methods, providing flexibility in customization.

# Key Features:
User-Friendly Interface: Built with Streamlit, the app offers an intuitive interface for users to upload query images, configure retrieval parameters, and visualize results.

Distance Metrics: Choose from a selection of distance metrics, including Euclidean, Canberra, Manhattan, Chebyshev, and Minkowsky, to tailor the similarity calculation.

Descriptor Methods: Select from different image descriptor methods such as bitdesc, GLCM, Haralick, or a combination of bitdesc and Haralick to extract features from images.

Normalization Option: Normalize image features for more robust and consistent similarity calculations.

Dynamic Result Visualization: The app displays the most similar images based on the user's query, providing a visual representation along with a frequency chart of unique classes.

# Getting Started:
Clone the repository.
Install the required dependencies: streamlit, cv2, numpy, PIL, scipy, scikit-learn.
Run the application using streamlit run app.py.
Explore and experiment with different configurations to enhance your image retrieval experience!

# Feel free to customize it according to any additional features or specifics about your project.
