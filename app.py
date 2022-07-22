import joblib
import pandas as pd
import streamlit as st
from PIL import Image

model = joblib.load('etc_trained_model.job')

image = Image.open('header.png')
image2 = Image.open('side.png')

hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
}
footer {
    visibility: visible;
}
footer:before{
    content:' © 2022 Krischan Kunkel,    '
}
</style>
"""
st.markdown(hide_menu,unsafe_allow_html=True)
st.header('Categorization of Medical Articles')
st.image(image)

with st.sidebar:
    st.subheader('Natural Language Processing')
    st.image(image2)
    st.write("""
    This model demonstrates **NLP** using a **Extreme Tree Classifier** -a very fast machine learning algorithm. 
    
    The purpose of the mode is to predict 20 medical categories from text inputs.
    The model was trained using approximately 65,000 articles from the New England Journal of Medicine. 

    """)

with st.expander("See explanation"):
     st.write("""
        Please enter the information of any medical article into the fields below to predict a corresponding category.
        You can also upload your own csv file with multiple entries to make predictions for several articles at once and then download the results.
     """)
file = st.file_uploader('Upload CSV File')

with st.expander("CSV Template (Copy & Paste)"):
    st.write('Copy the code below into an empty csv file...')
    st.code(
    """
    ,title,authors,item_text
    0,your title,the authors,"The document text." """
    )

if file:
    try:
        with st.spinner('Wait for it, making your predictions...'):
            df = pd.read_csv(file, index_col=0)
            df.reset_index(drop=True, inplace=True)
            df.insert(0, 'category_prediction', model.predict(df))
            st.dataframe(df)
            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='NLP-Predictions.csv',
                mime='text/csv',
            )
    except:
        st.error('Please make sure you are using the right file format. Use the template if necessary.')


if not file:
    st.write('Please enter the article information below:')
    with st.form('article_input', clear_on_submit=True):
        title = st.text_input(label='Title')
        authors = st.text_input(label='Authors')
        item_text = st.text_area(label='Abstract')
        submitted = st.form_submit_button('Predict Category')

    df = pd.DataFrame(
        {'title':[title], 
        'authors':[authors], 
        'item_text':[item_text]}
    )

    prediction = model.predict(df)[0]

    if submitted:
        if len(title) > 0:
            st.success(f'The predicted article category is {prediction}.')
        else:
            st.error('Please enter a title before trying to predict...')