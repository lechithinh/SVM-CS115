import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import pickle
import streamlit as st
from PIL import Image
import cv2



test_model = pickle.load(open("Classification_Model.p","rb"))

AVATAR = 'logo.jpg'
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
    st.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry standard dummy text ever since the")

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
    scale_percent = 60 # percent of original size
    width_pro= int(image_profile.shape[1] * scale_percent / 100)
    height_pro = int(image_profile.shape[0] * scale_percent / 100)
    dim = (width_pro, height_pro)
    resized_pro = cv2.resize(image_profile, dim, interpolation = cv2.INTER_AREA)
    st.image(resized_pro)

    st.markdown('''
          # ABOUT US \n 
           Let's call this web **SVM**. There are a plenty of features that this web can operate.\n
           
            Hacking AI could be classified into two groups: **AI Against Hacking** and **AI To Hack**. \n
            
            Here are our fantastic features:
            - Image Classification
        
            It can be acknowledged that no matter what Types of hacking AI are, these are one of the most basic things we should approach if heading to this field. \n
            Since these are just ideas, it would be better if we have more time to continue with our work. Promisingly, the next version would surely be well-structured and effective.
             
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
            st.image(resize(img_array,(350,350,3)))
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
        st.image(resize(image,(350,350,3)))
        
        
    
    

