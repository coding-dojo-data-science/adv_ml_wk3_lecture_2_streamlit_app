import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
from lime.lime_tabular import LimeTabularExplainer
import sys, os, json
sys.path.append(os.path.abspath('../'))
import functions as fn

# Open the json filepaths
with open('config/filepaths.json', 'r') as f:
    FPATHS = json.load(f)

# Define a function for loading the data
@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)

#Load train data
X_train, y_train = load_Xy_data(fpath = FPATHS['data']['ml']['train'])

# Load test data
X_test, y_test = load_Xy_data(fpath = FPATHS['data']['ml']['test'])

# Define function for loading model
def load_model_ml(fpath):
    return joblib.load(fpath)

@st.cache_resource
def get_explainer(_model_pipe, X_train, labels):
    X_train_sc = _model_pipe[0].transform(X_train)
    feature_names = _model_pipe[0].get_feature_names_out()
    explainer = LimeTabularExplainer(
                    X_train_sc,
                    mode='classification',
                    feature_names=feature_names,
                    class_names=labels,
                    random_state=42
                    )
    return explainer


@st.cache_resource
def explain_instance(_explainer, _model_pipe, instance_to_explain):
    instance_to_explain_sc = _model_pipe[0].transform(instance_to_explain)
    explanation = _explainer.explain_instance(instance_to_explain_sc[0],
                                             _model_pipe[-1].predict_proba
                                             )
    return explanation

st.title('Model Evaluation and Predictions')

# Add a selectbox for choosing the model
model_name = st.sidebar.selectbox('Select Model', ['logistic_regression', 'random_forest'], index = 0)

model = load_model_ml(FPATHS['models'][model_name])

# Evaluate the model subheader and button
st.subheader('Evaluation')

labels = ['Approved', 'Rejected']

#When button is pressed
if st.sidebar.button('Evaluate Model'):
    st.subheader(f'Evaluation of {model_name}')
    ## Evaluate the model
    train_report, test_report, eval_fig = fn.eval_classification(model, X_train, y_train, X_test, y_test, labels = labels )
    # Display the results
    st.text('Training Report')
    st.text(train_report)
    st.text('Testing Report')
    st.text(test_report)
    st.pyplot(eval_fig)


# Make prediction
st.sidebar.subheader('Make a prediction')

# Feature inputs
dependents = st.sidebar.slider('Number of Dependents', min_value=0, max_value=5)
graduated = st.sidebar.radio('Graduated College', ['Not Graduate', 'Graduate'])
self_employed = st.sidebar.radio('Self Employed', ['Yes', 'No'])
income_annum = st.sidebar.number_input('Annual Income', min_value=20000, max_value=10000000, step = 10000)
loan_term = st.sidebar.slider('Years to Repay', min_value=2, max_value=20)
cibil_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=900)
res_asset_val = st.sidebar.number_input('Home Value', min_value=10000, max_value=30000000)
comm_asset_val = st.sidebar.number_input('Value of Commercial Assets', min_value=1000, max_value=20000000)
lux_asset_val = st.sidebar.number_input('Value of Luxury Assets', min_value=0, max_value=40000000)
bank_asset_val = st.sidebar.number_input('Cash in Bank', min_value=0, max_value=20000000)
loan_amount = st.sidebar.number_input('Requested Loan Amount', min_value=300000, max_value=40000000)


explain = st.sidebar.checkbox('Explain Prediction')
predict = st.sidebar.button('Make Prediction')

if predict:
    try:
        ## Add inputs to a dataframe in order of original columns.  Must be 2 dimensional for model.
        user_data = pd.DataFrame([[dependents, graduated, self_employed, income_annum, loan_amount, loan_term, cibil_score, res_asset_val, comm_asset_val,
                                   lux_asset_val, bank_asset_val]], columns=X_train.columns)
        ## Get model prediction
        prediction = model.predict(user_data)[0]

        ## Change text color based on model prediction
        ## Red if loan is rejected, Green if it's accepted
        if prediction == 'Rejected':
            color = "red"
        else:
            color = "green"
        
        ## Print model prediction to page using html
        st.markdown(f'# <span style="color:{color}"> Loan {prediction} </span>',
                   unsafe_allow_html=True) # required for html in markdown
        
        ## If Explain Prediction checkbox checked
        if explain:
            explainer = get_explainer(model, X_train=X_train, labels=labels)
            explanation = explain_instance(explainer, model, user_data)
            components.html(explanation.as_html(show_predicted_value=False), 
                            height = 1000
                            )

    ## Catch errors
    except Exception as e:
        st.text(e)
        st.header('Please input all required information on the left')
























