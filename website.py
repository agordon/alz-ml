import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
import base64
import matplotlib.pyplot as plt 

fake_file = False

try:
    regdata = pd.read_csv("ADNI-data/regdata2.csv")
    regdata.drop(['Unnamed: 0'], axis = 1, inplace = True)
    meds = pd.read_csv("ADNI-data/med_data2.csv")
except FileNotFoundError:
    regdata = pd.read_csv("MMSE_fake.csv")
    meds = pd.read_csv("meds_fake.csv")
    fake_file = True
linear_regression = LinearRegression()
regdata1 = regdata.drop(['MMSE6',"MMSE12","MMSE18","MMSE24",'MMSE_slp'],axis=1)


st.set_page_config(
    page_title="MMSE Score Prediction",
)
if fake_file:
    st.title("Alzheimer's Disease Machine Learning Model (FAKE DATA)")
else:
    st.title("Alzheimer's Disease Machine Learning Model")


st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Our Research",'Data'])
input_dict = {}
values =[]
if page == "Home":
    st.subheader("Please enter the following information to the best of your ability:")
    st.write("Step 1: Basic Demographic Information")
    age = st.text_input("1. What is your age?")
    input_dict['Age'] = age
    gender = st.selectbox("2. What is your biological sex ?",['Male','Female'])
    if gender == 'Male':
        input_dict['Male'] = 1
    else:
        input_dict['Male'] = 0
    #edu = st.text_input("3. How many years of formal education have you received (including grade school)?")
    #input_dict['PTEDUCAT'] = edu
    if gender and age :
        st.markdown("---")
        st.write("Step 2: Medical Information")
        
        apoe = st.text_input("5. What is your APOE genotype?")
        input_dict['APOE_4'] = apoe.count('4')
        input_dict['APOE_2'] = apoe.count('2')
        dis = st.multiselect("6. Comorbidities",['Cardiovascular Issues','Head,Eyes,Nose,Ears, or Throat Issues','Muscoskeletal Issues','Gastrointestinal Issues','Psychiatric Issues','None'])
        if 'Cardiovascular Issues' in dis:
            input_dict['card_issues'] = 1
        if 'Psychiatric Issues' in dis:
            input_dict['psych_issues'] = 1
        if dis and apoe:
            st.markdown("---")
            st.write("Step 3: Cognitive Score Prediction") 
            mmse1 = st.text_input("8. Most recent MMSE Score")
            input_dict['MMSE_baseline'] = mmse1 
            time = int(st.radio("9. Predict score after how many months?",[6,12,24]))
            #if mmse1:
            if mmse1 and time:
                st.markdown("---")
                for col in regdata1.drop('PTID',axis=1).columns:
                    if col in input_dict:
                        values.append(int(input_dict[col]))
                    else:
                        values.append(0)
                linear_regression.fit(regdata.dropna(subset=[f'MMSE{str(time)}']).drop(['MMSE6',"MMSE12","MMSE18","MMSE24",'MMSE_slp','PTID'],axis=1), regdata[[f'MMSE{str(time)}']].dropna())
                score = round(float(linear_regression.predict(np.array([values]))[0][0]))
                st.metric("Predicted Score:", score,score-int(mmse1))

                x = [6, 12, 24]
                y=[]
                for val in x:
                    linear_regression.fit(regdata.dropna(subset=[f'MMSE{str(val)}']).drop(['MMSE6',"MMSE12","MMSE18","MMSE24",'MMSE_slp','PTID'],axis=1), regdata[[f'MMSE{str(val)}']].dropna())
                    y.append(round(float(linear_regression.predict(np.array([values]))[0][0])))
                fig, ax = plt.subplots()
                ax.plot(x, y, label="MMSE")
                ax.set_xlabel('Time(months)')
                ax.set_ylabel('MMSE Score')
                ax.set_title('Predicted MMSE Trajectory')
                ax.legend()

                # 4. Display the plot in Streamlit
                st.pyplot(fig)
                #score = decline+int(mmse1)
                #st.line_chart(data)
                regdata_train = regdata1.merge(meds,on='PTID', how = 'left')
                regdata_train['med_none'] = regdata_train['med_donepezil'].isna().astype(int)
                regdata_train = regdata_train[(regdata_train['med_donepezil']!=1) | (regdata_train['med_memantine']!=1)]
                regdata_train.fillna(0,inplace=True)
                slope_dict = {}
                
                for treatment in ['med_galantamine',"med_memantine",'med_rivastigmine','med_donepezil']:
                    log_x=regdata_train.drop(['PTID','med_galantamine',"med_memantine",'med_rivastigmine','med_donepezil'],axis=1)
                    log_x = pd.concat([log_x,regdata_train[treatment]],axis=1)
                    log_x = log_x[(log_x[treatment]==1) | (log_x['med_none']==0)]
                    log_x = log_x.drop(['med_none'],axis=1)
                    log_x[treatment] = log_x[treatment].replace([2,3,4,5,6,7,8], 1)
                    clf = LogisticRegression(random_state=0).fit(log_x.drop([treatment],axis=1), log_x[treatment])
                    log_x['propensity'] = clf.predict_proba(log_x.drop([treatment],axis=1))[:, 1]
                    new_prop = clf.predict_proba(np.array(values).reshape(1, -1))[:, 1][0]

                    treated = log_x[log_x[treatment]==1]#.drop('drop',axis=1)

                    nn = NearestNeighbors(n_neighbors=4,radius=0.05, metric='euclidean')
                    nn.fit(treated[['propensity']])
                    distances, indices = nn.kneighbors(np.array([[new_prop]]))



                    #valid = distances.flatten() <= 0.001

                    #treated_matched = treated[valid].reset_index(drop=True)
                    #matched_control = control.iloc[list(indices.flatten())]
                    #matched_control = matched_control[valid].reset_index(drop=True)
                    
                    #caliper = 0.05
                    #valid = distances.flatten() <= caliper
                    #treated_matched = treated[valid].reset_index(drop=True)
                    #matched_control = control.iloc[list(indices.flatten())][valid].reset_index(drop=True)
                    
                    #treated = treated.reset_index(drop=True)
                    #matched_control = control.iloc[list(indices.flatten())]
                    #matched_control = matched_control.reset_index(drop=True)
                    slope_dict[treatment]=int(regdata.iloc[indices[0],[6]].mean())
                st.metric("Reccomended Treatment:", max(slope_dict, key=slope_dict.get)[4:].capitalize())
elif page == "Our Research":

    def display_pdf(file_path):
        # 1. Read the file as binary
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="900" height="1000" type="application/pdf"></iframe>'

        st.markdown(pdf_display, unsafe_allow_html=True)

    display_pdf("CYSF 2026 Paper_ Comparing Longitudinal Effectiveness of Existing Alzheimer’s Disease Treatments.pdf")
else:
    st.subheader("Data Collection and Methods")
    st.write('''We used ADNI data to train our machine learning model. The Alzheimer's Disease Neuroimaging Initiative is a longutitudinal study that has collected data
             for many years in order to capture long term trends in Alzheimer's disease progression. It is devoted to providing researchers around the world with 
             free, real-world, patient level data in order to expand the field of neuroscience. In order to make this model, ADNI data was cleaned and the needed features were
             isolated and trained on a regression model. Afterwards, propensity score matching was utilized in order to ensure a fair comparison of the control and treated groups 
             in the training dataset. This method matches treated patients with similar, untreated patients and compares the cognitive scores of both. This was done for every treatment. 
             When a new patient enters their data, they are fitted on the trained model in order to predict their future CDR-SB score. They are also matched up with similar treated patients in
             order to recommend a certain Alzheimer's treatment.
              ''')

st.markdown("---")
st.caption("By: Taha Farooq & Manan Vyas")
st.caption("11th grade")
st.caption("Renert School")
st.caption("NOT FOR CLINICAL ADVICE")