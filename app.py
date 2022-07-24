import joblib
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px

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
.block-container{ padding-top: 10px}
</style>
"""
st.markdown(hide_menu,unsafe_allow_html=True)
st.header('Categorization of Medical Articles')
st.subheader('Using Machine Learning - NLP')
st.image(image)

with st.sidebar:
    st.subheader('Natural Language Processing')
    st.image(image2)
    st.write("""
    This model demonstrates the usage of **NLP** by utilizing the **Extreme Tree Classifier** machine learning algorithm. 
    
    The purpose of the model is to predict 20 medical categories from medical articles by using Natural Language Processing (NLP).
    
    The model was trained using approximately 65,000 articles from the New England Journal of Medicine. 

    """)

with st.expander("See explanation"):
     st.write("""
        Please enter the information of any medical article into the fields below to predict a corresponding category.
        You can also upload your own csv file with multiple entries to make predictions for several articles at once and then download the results.
     """)
file = st.file_uploader('Upload CSV File')
demo = st.button(label='Demo Data')

with st.expander("CSV Template (Copy & Paste)"):
    st.write('Copy the code below into an empty csv file...')
    st.code(
    """
    ,title,authors,item_text
    0,your title,the authors,"The document text." """
    )

if demo:
    with st.spinner('Wait for it, making your predictions...'):
            df = pd.read_csv('ExampleData.csv', index_col=0)
            df.reset_index(drop=True, inplace=True)
            df.insert(0, 'category_prediction', model.predict(df))
            st.dataframe(df)
            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            fig = px.bar(        
            df['category_prediction'].value_counts(),
            title = "Predicted Categories",
            orientation = 'h' #Optional Parameter
            )
            st.plotly_chart(fig)


            csv = convert_df(df)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.download_button(
                    label="Download CSV File",
                    data=csv,
                    file_name='NLP-Predictions.csv',
                    mime='text/csv',
                )
            with col2:
                clear = st.button('Clear')
                if clear:
                    demo = False

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
            fig = px.bar(        
            df['category_prediction'].value_counts(),
            title = "Predicted Categories",
            orientation = 'h' #Optional Parameter
            )
            st.plotly_chart(fig)
            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='NLP-Predictions.csv',
                mime='text/csv',
            )
            
    except:
        st.error('Please make sure you are uploading a valid a csv file. Use the template if necessary.')


if not demo and not file:
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