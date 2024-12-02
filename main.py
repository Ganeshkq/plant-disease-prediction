# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import time
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import streamlit as st

# # TensorFlow Model Prediction
# def model_prediction(test_image):
#     model = load_model("trained_model.keras")
#     image = Image.open(test_image)
#     image = image.resize((128, 128))
#     input_arr = np.array(image) / 255.0  # Normalize the image
#     input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
#     predictions = model.predict(input_arr)
#     return np.argmax(predictions)  # Return index of max element

# # Scanning Effect
# def scanning_effect(image_path):
#     image = Image.open(image_path)
#     width, height = image.size
#     step = height // 1  # Number of steps for scanning effect
#     for i in range(0, height, step):
#         if i + step > height:
#             step = height - i
#         img_part = image.crop((0, 0, width, i + step))
#         st.image(img_part, use_column_width=True)
#         time.sleep(0.1)  # Delay for scanning effect

# # Plant Disease Information Dictionary
# plant_info = {
#     'Apple___Apple_scab': "Apple scab is a disease caused by the fungus Venturia inaequalis. It affects the leaves and fruit of apple trees.",
#     'Apple___Black_rot': "Black rot, caused by the fungus Botryosphaeria obtusa, can affect both fruit and leaves, causing dark, sunken spots.",
#     'Apple___Cedar_apple_rust': "Cedar apple rust is a fungal disease that causes bright orange spots on leaves and fruit.",
#     'Apple___healthy': "No disease detected. The apple is healthy.",
#     'Blueberry___healthy': "No disease detected. The blueberry is healthy.",
#     'Cherry_(including_sour)___Powdery_mildew': "Powdery mildew is a fungal disease that covers leaves with a white, powdery coating.",
#     'Cherry_(including_sour)___healthy': "No disease detected. The cherry is healthy.",
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Gray leaf spot is a fungal disease that causes gray to tan spots on leaves.",
#     'Corn_(maize)___Common_rust_': "Common rust is a fungal disease that produces reddish-brown pustules on leaves.",
#     'Corn_(maize)___Northern_Leaf_Blight': "Northern leaf blight causes elongated grayish-green lesions on leaves.",
#     'Corn_(maize)___healthy': "No disease detected. The corn is healthy.",
#     'Grape___Black_rot': "Black rot causes black, shriveled fruit and brown spots on leaves.",
#     'Grape___Esca_(Black_Measles)': "Esca, also known as black measles, causes dark streaks in wood and leaf discoloration.",
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Leaf blight causes dark spots on leaves and can lead to leaf drop.",
#     'Grape___healthy': "No disease detected. The grape is healthy.",
#     'Orange___Haunglongbing_(Citrus_greening)': "Citrus greening is a bacterial disease that causes yellowing of leaves and misshapen fruit.",
#     'Peach___Bacterial_spot': "Bacterial spot causes dark, water-soaked spots on leaves and fruit.",
#     'Peach___healthy': "No disease detected. The peach is healthy.",
#     'Pepper,_bell___Bacterial_spot': "Bacterial spot affects peppers, causing small, water-soaked spots on leaves and fruit.",
#     'Pepper,_bell___healthy': "No disease detected. The bell pepper is healthy.",
#     'Potato___Early_blight': "Early blight causes dark, concentric rings on leaves and tubers.",
#     'Potato___Late_blight': "Late blight is a serious disease that causes dark lesions on leaves, stems, and tubers.",
#     'Potato___healthy': "No disease detected. The potato is healthy.",
#     'Raspberry___healthy': "No disease detected. The raspberry is healthy.",
#     'Soybean___healthy': "No disease detected. The soybean is healthy.",
#     'Squash___Powdery_mildew': "Powdery mildew causes a white, powdery coating on leaves.",
#     'Strawberry___Leaf_scorch': "Leaf scorch causes brown edges on leaves and can reduce plant vigor.",
#     'Strawberry___healthy': "No disease detected. The strawberry is healthy.",
#     'Tomato___Bacterial_spot': "Bacterial spot causes small, dark, water-soaked spots on leaves and fruit.",
#     'Tomato___Early_blight': "Early blight causes dark, concentric rings on leaves and fruit.",
#     'Tomato___Late_blight': "Late blight is a devastating disease that causes dark, water-soaked lesions on leaves and fruit.",
#     'Tomato___Leaf_Mold': "Leaf mold causes pale green or yellow spots on the upper side of leaves and gray mold on the underside.",
#     'Tomato___Septoria_leaf_spot': "Septoria leaf spot causes small, circular spots on leaves, leading to defoliation.",
#     'Tomato___Spider_mites Two-spotted_spider_mite': "Two-spotted spider mites cause yellowing and stippling of leaves.",
#     'Tomato___Target_Spot': "Target spot causes small, dark, target-like spots on leaves and fruit.",
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "TYLCV causes yellowing and curling of leaves and stunted growth.",
#     'Tomato___Tomato_mosaic_virus': "Tomato mosaic virus causes mottling and distortion of leaves.",
#     'Tomato___healthy': "No disease detected. The tomato is healthy."
# }

# # Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About us", "Disease Recognition", "Plant Village", "Chat With Specialist", "E - Nursery","Contact us"])

# # Main Page
# if app_mode == "Home":
#     st.header("PLANT DISEASE RECOGNITION SYSTEM")
#     image_path = "home_page.jpeg"
#     st.image(image_path, use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍
    
#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)

# elif app_mode == "About us":
#     st.header("About us")
#     st.markdown("""
#     #### About Dataset
#     This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
#     This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
#     A new directory containing 33 test images is created later for prediction purposes.
    
#     #### Content
#     1. train (70,295 images)
#     2. test (33 images)
#     3. validation (17,572 images)
#     """)

# elif app_mode == "Disease Recognition":
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
#     if test_image:
#         st.write("Scanning Image...")
#         scanning_effect(test_image)
        
#         if st.button("Predict"):
#             st.snow()
#             st.write("Our Prediction")
#             try:
#                 result_index = model_prediction(test_image)
#                 # Reading Labels
#                 class_name = [
#                     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
#                     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
#                     'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
#                     'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
#                     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
#                     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
#                     'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
#                     'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
#                     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
#                     'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                     'Tomato___healthy'
#                 ]
#                 st.success("Model is Predicting it's a {}".format(class_name[result_index]))
#                 st.info(plant_info.get(class_name[result_index], "No information available for this disease."))
#             except Exception as e:
#                 st.error("Error occurred: {}".format(e))


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = load_model("trained_model.keras")  # Correct model loading function
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Plant Disease Information Dictionary
plant_info = {
    'Apple___Apple_scab': "Apple scab is a disease caused by the fungus Venturia inaequalis. It affects the leaves and fruit of apple trees.",
    'Apple___Black_rot': "Black rot, caused by the fungus Botryosphaeria obtusa, can affect both fruit and leaves, causing dark, sunken spots.",
    'Apple___Cedar_apple_rust': "Cedar apple rust is a fungal disease that causes bright orange spots on leaves and fruit.",
    'Apple___healthy': "No disease detected. The apple is healthy.",
    'Blueberry___healthy': "No disease detected. The blueberry is healthy.",
    'Cherry_(including_sour)___Powdery_mildew': "Powdery mildew is a fungal disease that covers leaves with a white, powdery coating.",
    'Cherry_(including_sour)___healthy': "No disease detected. The cherry is healthy.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Gray leaf spot is a fungal disease that causes gray to tan spots on leaves.",
    'Corn_(maize)___Common_rust_': "Common rust is a fungal disease that produces reddish-brown pustules on leaves.",
    'Corn_(maize)___Northern_Leaf_Blight': "Northern leaf blight causes elongated grayish-green lesions on leaves.",
    'Corn_(maize)___healthy': "No disease detected. The corn is healthy.",
    'Grape___Black_rot': "Black rot causes black, shriveled fruit and brown spots on leaves.",
    'Grape___Esca_(Black_Measles)': "Esca, also known as black measles, causes dark streaks in wood and leaf discoloration.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Leaf blight causes dark spots on leaves and can lead to leaf drop.",
    'Grape___healthy': "No disease detected. The grape is healthy.",
    'Orange___Haunglongbing_(Citrus_greening)': "Citrus greening is a bacterial disease that causes yellowing of leaves and misshapen fruit.",
    'Peach___Bacterial_spot': "Bacterial spot causes dark, water-soaked spots on leaves and fruit.",
    'Peach___healthy': "No disease detected. The peach is healthy.",
    'Pepper,_bell___Bacterial_spot': "Bacterial spot affects peppers, causing small, water-soaked spots on leaves and fruit.",
    'Pepper,_bell___healthy': "No disease detected. The bell pepper is healthy.",
    'Potato___Early_blight': "Early blight causes dark, concentric rings on leaves and tubers.",
    'Potato___Late_blight': "Late blight is a serious disease that causes dark lesions on leaves, stems, and tubers.",
    'Potato___healthy': "No disease detected. The potato is healthy.",
    'Raspberry___healthy': "No disease detected. The raspberry is healthy.",
    'Soybean___healthy': "No disease detected. The soybean is healthy.",
    'Squash___Powdery_mildew': "Powdery mildew causes a white, powdery coating on leaves.",
    'Strawberry___Leaf_scorch': "Leaf scorch causes brown edges on leaves and can reduce plant vigor.",
    'Strawberry___healthy': "No disease detected. The strawberry is healthy.",
    'Tomato___Bacterial_spot': "Bacterial spot causes small, dark, water-soaked spots on leaves and fruit.",
    'Tomato___Early_blight': "Early blight causes dark, concentric rings on leaves and fruit.",
    'Tomato___Late_blight': "Late blight is a devastating disease that causes dark, water-soaked lesions on leaves and fruit.",
    'Tomato___Leaf_Mold': "Leaf mold causes pale green or yellow spots on the upper side of leaves and gray mold on the underside.",
    'Tomato___Septoria_leaf_spot': "Septoria leaf spot causes small, circular spots on leaves, leading to defoliation.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Two-spotted spider mites cause yellowing and stippling of leaves.",
    'Tomato___Target_Spot': "Target spot causes small, dark, target-like spots on leaves and fruit.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "TYLCV causes yellowing and curling of leaves and stunted growth.",
    'Tomato___Tomato_mosaic_virus': "Tomato mosaic virus causes mottling and distortion of leaves.",
    'Tomato___healthy': "No disease detected. The tomato is healthy."

}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About us", "Disease Recognition", "Plant Village", "Chat With Specialist", "E - Nursery", "Contact us"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About us":
    st.header("About us")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.
    
    #### Content
    1. train (70,295 images)
    2. test (33 images)
    3. validation (17,572 images)
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    if test_image:
        st.write("Displaying Image...")
        # Directly display the uploaded image
        image = Image.open(test_image)
        st.image(image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            try:
                result_index = model_prediction(test_image)
                # Reading Labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                st.success("Model is Predicting it's a {}".format(class_name[result_index]))
                st.info(plant_info.get(class_name[result_index], "No information available for this disease."))
            except Exception as e:
                st.error("Error occurred: {}".format(e))

# Add other pages for Plant Village, Chat, E-Nursery, and Contact Us as needed.
