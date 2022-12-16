import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import pickle
import streamlit as st
from PIL import Image
import cv2



test_model = pickle.load(open("Classification_Model.p","rb"))

AVATAR = 'avatar.png'
demo_image = 'dog.jpg'
Categories = ["cat","dog"]

st.title('CS115 | IMAGE CLASSIFICATION SVM')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.title('Workspace - Options')
st.sidebar.subheader('Parameters')



app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['General information',
                                    'Image Classfication']
                                )

if app_mode == 'General information':
    st.markdown("**Support Vector Machine (SVM)** is one of the most popular Machine Learning Classifier. It falls under the category of Supervised learning algorithms and uses the concept of Margin to classify between classes.")

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    
    image_profile = np.array(Image.open(AVATAR))
    scale_percent = 200 # percent of original size
    width_pro= int(image_profile.shape[1] * scale_percent / 100)
    height_pro = int(image_profile.shape[0] * scale_percent / 100)
    dim = (width_pro, height_pro)
    resized_pro = cv2.resize(image_profile, dim, interpolation = cv2.INTER_AREA)
    st.image(resized_pro)

    st.markdown('''
          # ABOUT US \n 
           Let's call this web **SVM IN PRACTISE**. There are a plenty of features that this web can operate.\n
            
            Here are our fantastic features:
            - Image Classification 
        
            Since this is just one sample, it would be better if we have more time to continue with our work. Promisingly, the next version would surely be well-structured and effective. \n
            We would implement more features in the next versions, feel free to contact and collaberate with us
            ''')
    
elif app_mode == 'Image Classfication':

    st.sidebar.markdown('---')

    st.markdown(
    """
    # Dog and Cat Classification
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.sidebar.text('Predicted Image')
    else:
        image = np.array(Image.open(demo_image))
        

    flat_data = []
    url = st.text_input('Enter url of image to test')
    if url: 
        #bug: size 
        try:
            img_array = imread(url)
            img_resized = resize(img_array,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data = np.array(flat_data)
            y_output = test_model.predict(flat_data)
            y_output = Categories[y_output[0]]
            st.markdown(f"**Predicted Result: {y_output.upper()}**")
            col1, col2, col3 = st.columns([1,3,1])
            with col1:
                st.write("")
            with col2:
                st.image(resize(img_array,(350,350,3)))
            with col3:
                st.write("")
        except:
            st.markdown(f"**Image is not accessible. Please try another one!**")
    else:
        st.sidebar.text('Sample Image')
        st.sidebar.image(image)
        img_resized = resize(image,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        y_output = test_model.predict(flat_data)
        y_output = Categories[y_output[0]]
        st.markdown(f"**Predicted Result: {y_output.upper()}**")
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            st.write("")
        with col2:
            st.image(resize(image,(350,350,3)))
        with col3:
            st.write("")
       
        
        
    
    

