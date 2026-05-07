import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import base64
import matplotlib.pyplot as plt 

fake_file = False

try:
    regdata = pd.read_csv("ADNI-data/prediction_data2.csv")
    regdata.drop(['Unnamed: 0'], axis = 1, inplace = True)
    meds = pd.read_csv("ADNI-data/med_data3.csv")
except FileNotFoundError:
    regdata = pd.read_csv("MMSE_fake.csv")
    meds = pd.read_csv("meds_fake.csv")
    fake_file = True
regressor = RandomForestRegressor(random_state = 200,max_features=4, min_samples_split = 15,max_depth = 6,n_estimators = 80) 
classifier = RandomForestClassifier(random_state = 108,max_features=3, min_samples_split = 10,max_depth = 5,n_estimators = 150,oob_score=True, min_samples_leaf = 5,class_weight='balanced_subsample')

regdata1 = regdata.drop(['MMSE6',"MMSE12","MMSE24",'Kclusters'],axis=1)
regdata = regdata[~regdata['PTID'].isin(['136_S_0873','007_S_0293','021_S_0424'])]

st.set_page_config(
    page_title="ADAS-Cog Score Prediction",
)
if fake_file:
    st.title("Alzheimer's Disease Machine Learning Model (FAKE DATA)")
else:
    st.title("Alzheimer's Disease Machine Learning Model")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predictive Model",'Data and Methods','Our Meta-Analysis'])
input_dict = {}
values =[]
query_params = st.query_params


if page == "Predictive Model":
    st.write('''This website predicts the Alzheimer's Disease Assesment Scale-Cognitive Subscale (ADAS-Cog) score of an Alzheimer's patient at a future
         point in time, taking current information into consideration. ADAS-Cog 13 is a 0-85 point cognitive measurement that evaluates 
            cognitive functioning through performance in memory, orientation, and language.
             This website also predicts a cognitive trajectory over time and recommends a treatment by matching the patient with treated patients
             based on other parameters and comparing their scores. The treatment group with the lowest score is recommended to the patient. 
''')
    st.subheader("Please enter the following information to the best of your ability:")
    st.write('Note: Remember to click enter after filling out each entry.')
    st.write("**Step 1: Basic Demographic Information**")
    age = st.text_input("1. What is the patient's age?")
    input_dict['Age'] = age
    gender = st.selectbox("2. What is the patient's biological sex?",['Male','Female'])
    if gender == 'Male':
        input_dict['Male'] = 1
    else:
        input_dict['Male'] = 0

    edu = st.text_input("3. How many years of formal education has the patient received (including grade school)?")
    input_dict['PTEDUCAT'] = edu
    if gender and age:
        #st.session_state['input'] = input_dict
    #if ('input' in st.session_state.keys()) and ('Male' in st.session_state['input'].keys()) and ('Age' in st.session_state['input'].keys()):
        st.markdown("---")
        st.write("**Step 2: Medical Information**")
        
        # 2. The Input
        apoe = st.radio(
            "4. What is the patient's APOE genotype?",
            ['2/2','2/3','2/4','3/3','3/4','4/4','I do not know'],
            help = "The APOE gene influences Alzheimer's disease progression. The '2' variation is preventative, '3' is neutral, and '4' is predisposing."
        )
        input_dict['APOE_4'] = '4' in apoe
        
        dis = st.radio("5. Does the patient have cardiovascular issues:",['Yes','No'])

        if 'Yes' in dis:
            input_dict['card_issues'] = 1
        else:
            input_dict['card_issues'] = 0

        brain = st.text_input("6. What is the patient's total brain volume?")
        input_dict['TOTAL_BRAIN'] = brain

        if dis and apoe:
         #   st.session_state['input'] = input_dict
        #if ('APOE_4' in st.session_state['input'].keys()) and ('card_issues' in st.session_state['input'].keys()): #and ('card_issues' in st.session_state['input'].keys()):
            st.markdown("---")
            st.write("**Step 3: Cognitive Score Prediction**") 
            mmse1 = st.text_input("7. Most recent ADAS-Cog Score")
            input_dict['MMSE_baseline'] = mmse1 
            time = int(st.radio("8. Predict score after how many months?",[6,12,24]))
            #if mmse1:
            if time and mmse1:
                #st.session_state['input'] = input_dict
                #st.session_state['time'] = time
            #if ('time' in st.session_state.keys()) and ('MMSE_baseline' in st.session_state['input'].keys()):
                st.markdown("---")
                for col in regdata1.drop(['PTID','Male','psych_issues'],axis=1).columns:
                    if col in input_dict:
                        values.append(float(input_dict[col]))
                    else:
                        values.append(0)
                regressor.fit(regdata.dropna(subset=[f'MMSE{str(time)}']).drop(['MMSE6',"MMSE12","MMSE24",'Male','psych_issues','Kclusters','PTID'],axis=1), regdata[[f'MMSE{str(time)}']].dropna())
                score = round(float(regressor.predict(np.array([values]))[0])*2)/2
                if score <0:
                    score = 0
                elif score >85:
                    score = 85
                st.metric("Predicted Score:", str(score)+' Points','Change: ' +str(round((score -float(mmse1))*2)/2),delta_color="inverse")

                values1 = []
                for col in regdata1.drop(['PTID'],axis=1).columns:
                    if col in input_dict:
                        values1.append(float(input_dict[col]))
                    else:
                        values1.append(0)
                classifier.fit(regdata.drop(['MMSE6',"MMSE12","MMSE24",'Kclusters','PTID'],axis=1), regdata[['Kclusters']])
                score = classifier.predict(np.array([values1]))[0]
        
                st.metric("Decline Rate Category:", str(score)+' Decliner')

                x = [0, 6, 12, 24]
                y=[]
                err = [0]
                for val in ['_baseline', 6, 12, 24]:
                    if type(val) == str:
                        y.append(float(mmse1))
                    else:
                        regressor.fit(regdata.dropna(subset=[f'MMSE{str(val)}']).drop(['MMSE6',"MMSE12","MMSE24",'Male','psych_issues','Kclusters','PTID'],axis=1), regdata[[f'MMSE{str(val)}']].dropna())
                        score = round(float(regressor.predict(np.array([values]))[0]))
                        if score <0:
                            score = 0
                        elif score >85:
                            score = 85
                        y.append(score)
                        if val != '_baseline':
                            tree_preds = np.array([tree.predict(np.array([values]))[0] for tree in regressor.estimators_])
                            interval = np.std(tree_preds, axis=0)*0.196
                            err.append(interval)
                fig, ax = plt.subplots()
                ax.errorbar(x, y,yerr=err, label="ADAS-Cog")
                ax.set_ylim(0,max(y)+8)
                ax.set_yticks(np.arange(0,max(y)+8,4))
                #print(list(np.arange(float(mmse1),y[-1]+1,0.5)))
                
                ax.set_xlabel('Time(months)')
                ax.set_ylabel('ADAS-Cog Score')
                ax.set_title('Predicted ADAS-Cog Trajectory')
                ax.legend()
                st.pyplot(fig)
                #score = decline+int(mmse1)
                #st.line_chart(data)


                regdata_train = regdata1.merge(meds,on='PTID', how = 'left')
                regdata_train['med_none'] = regdata_train['med_donepezil'].isna().astype(int)
                regdata_train = regdata_train[(regdata_train['med_donepezil']!=1) | (regdata_train['med_memantine']!=1)]
                regdata_train.fillna(0,inplace=True)
                slope_dict = {}
                
                values = []
                for col in regdata1.drop(['PTID'],axis=1).columns:
                    if col in input_dict:
                        values.append(float(input_dict[col]))
                    else:
                        values.append(0)
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
                    slope_dict[treatment]=float(regdata.iloc[indices[0],[3]].mean())
                st.metric("Recommended Treatment is:", min(slope_dict, key=slope_dict.get)[4:].capitalize())
                st.write("which was predicted to be the treatment that minimizes ADAS-Cog score for similar subtypes of patients.")
elif page == "Our Meta-Analysis":

    def display_pdf(file_path):
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        user_agent = st.context.headers.get("User-Agent", "").lower()
        is_mobile = ("iphone" in user_agent 
                    or "android" in user_agent 
                    or "mobile" in user_agent)

        if is_mobile:
            
            st.warning("Your device cannot display PDFs inline. Please download it instead:")
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="document.pdf",
                mime="application/pdf"
            )
        else:
            
            pdf_html = f"""
                <iframe 
                    src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" 
                    height="800px">
                </iframe>
            """
            st.markdown(pdf_html, unsafe_allow_html=True)
    display_pdf("CYSF 2026 Paper_ Comparing Longitudinal Effectiveness of Existing Alzheimer’s Disease Treatments.pdf")
else:
    st.subheader("Data Collection and Methods")
    st.write('''We used Alzheimer's Disease Neuroimaging Initiative( ADNI) data to train our machine learning model. ADNI is a longutitudinal study that has collected data
             for many years in order to capture long term trends in Alzheimer's disease progression. It is devoted to providing researchers around the world with 
             free, real-world, patient level data in order to expand the field of neuroscience. In order to make this model, ADNI data was cleaned and the needed features were
             isolated and trained on a regression model. Afterwards, propensity score matching was utilized in order to ensure a fair comparison of the control and treated groups 
             in the training dataset. This method matches treated patients with similar, untreated patients and compares the cognitive scores of both. This was done for every treatment. 
             When a new patient enters their data, they are fitted on the trained model in order to predict their future ADAS-Cog score. They are also matched up with similar treated patients in
             order to recommend a certain Alzheimer's treatment.
              ''')

st.markdown("---")
st.caption("By: Taha Farooq & Manan Vyas")
st.caption("11th grade")
st.caption("Renert School")
st.caption("NOT FOR CLINICAL ADVICE")