import mlflow
import time
import streamlit as st

st.set_page_config(page_title = 'WineQ Prediction',
                    page_icon = '🦅', 
                    layout = 'centered',
                    initial_sidebar_state = 'expanded') 

# Sidebar Infor
st.sidebar.title("About Me 🤖")
try:
     st.sidebar.header(f"Model Name\n ```{'model.name'}```")
     st.sidebar.header(f"Model Version\n ```{'model.version'}```")
     st.sidebar.header(f"Current Stage\n ```{'model.current_stage'}```")
     st.sidebar.subheader(f"Run ID\n ```{'model.run_id'}```")
     # with st.sidebar.progress():
     if 'loaded_model' not in st.session_state:        
          with st.spinner('Loading Models') : 
               time.wait(10)  
          #   st.session_state['loaded_model'] = mlflow.pyfunc.load_model(logged_model)
          #   st.session_state['loaded_vect']= mlflow.sklearn.load_model(logged_vect)
     st.sidebar.success("Server is Up & Models are loaded 🔥")
except :
     st.sidebar.warning("Models not found")


#  Main Area   
st.title("Wine Quality 🍷")    
# Making Predictions 
text = st.text_area("Enter your Message / Email in the box below 👇")
if st.button("Predict 🚀") :
     print()  
#     processed_text = preprocess_text(text)
#     vectorized_text = st.session_state['loaded_vect'].transform([processed_text])
#     prediction = st.session_state['loaded_model'].predict(vectorized_text)
#     if prediction[0]=='spam':
     #    st.subheader(f"Looks like a Spam ❌")
#     else:
     #    st.subheader(f"Looks Safe ✅")

