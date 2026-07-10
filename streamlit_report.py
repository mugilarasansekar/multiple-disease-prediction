# import packages
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Streamlit
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")
st.title("Multiple Disease Prediction")
st.caption("Fill in the patient details below to check the prediction for a disease.")
st.divider()

# Load model files
@st.cache_resource
def load_model_files():
    disease = ["parkinsons", "liver", "kidney"]
    models = ["lr", "rf", "xg"]
    files = {}
    path = "./models/"
    for i in disease:
        for j in models:
            name = i+"_"+j
            files[name] = joblib.load(path+name+"_model.pkl")
        files[i+"_scaler"] = joblib.load(path+i+"_scaler.pkl")
        files[i+"_features"] = joblib.load(path+i+"_features.pkl")
    
    return files

# Load model files
files = load_model_files()

# Parkinsons Disease Form
def parkinsons_form():
    st.subheader("Parkinsons Disease - Patient Details")
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", 80.0, 270.0, 200.0)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 300.0)
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", 60.0, 240.0, 180.0)
        mdvp_jitter_pct = st.number_input("MDVP:Jitter(%)", 0.0, 0.05, 0.002)
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.001, 0.00004)
        mdvp_rap = st.number_input("MDVP:RAP", 0.0, 0.03, 0.003)
        mdvp_ppq = st.number_input("MDVP:PPQ", 0.0, 0.02, 0.003)
        jitter_ddp = st.number_input("Jitter:DDP", 0.0, 0.07, 0.01)
    with col2:
        mdvp_shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.13, 0.03)
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 1.5, 0.28)
        shimmer_apq3 = st.number_input("Shimmer:APQ3", 0.0, 0.06, 0.016)
        shimmer_apq5 = st.number_input("Shimmer:APQ5", 0.0, 0.08, 0.018)
        mdvp_apq = st.number_input("MDVP:APQ", 0.0, 0.15, 0.024)
        shimmer_dda = st.number_input("Shimmer:DDA", 0.0, 0.17, 0.047)
        nhr = st.number_input("NHR", 0.0, 0.35, 0.025)
    with col3:
        hnr = st.number_input("HNR", 5.0, 35.0, 22.0)
        rpde = st.number_input("RPDE", 0.2, 0.7, 0.5)
        d2 = st.number_input("D2", 1.0, 4.0, 2.4)
        dfa = st.number_input("DFA", 0.5, 0.85, 0.72)
        spread1 = st.number_input("Spread1", -8.0, -2.0, -5.7)
        spread2 = st.number_input("Spread2", 0.0, 0.5, 0.23)
        ppe = st.number_input("PPE", 0.0, 0.55, 0.21)

    input_data = {
        "MDVP:Fo(Hz)": mdvp_fo,
        "MDVP:Fhi(Hz)": mdvp_fhi,
        "MDVP:Flo(Hz)": mdvp_flo,
        "MDVP:Jitter(%)": mdvp_jitter_pct,
        "MDVP:Jitter(Abs)": mdvp_jitter_abs,
        "MDVP:RAP": mdvp_rap,
        "MDVP:PPQ": mdvp_ppq,
        "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": mdvp_shimmer,
        "MDVP:Shimmer(dB)": mdvp_shimmer_db,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": mdvp_apq,
        "Shimmer:DDA": shimmer_dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "D2": d2,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "PPE": ppe,
    }
    return input_data

# Liver Disease input form
def liver_form():
    st.subheader("Liver Disease - Patient Details")
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 4, 90, 45)
        gender = st.selectbox("Gender", ("Male", "Female"))
        total_bilirubin = st.number_input("Total Bilirubin", 0.1, 80.0, 1.0)
    with col2:
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase", 60.0, 2500.0, 290.0)
        almaine_aminotransferase = st.number_input("Almaine Aminotransferase", 10.0, 2000.0, 80.0)
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", 10.0, 5000.0, 100.0)
    with col3:
        total_protiens = st.number_input("Total Protiens", 2.0, 10.0, 3.0)
        albumin = st.number_input("Albumin", 0.1, 6.0, 1.0)
        albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", 0.1, 6.0, 1.0)

    # Encode Gender
    if gender == "Male":
        gender_encoded = 1
    else:
        gender_encoded = 0
    
    # Log Transform
    log_values = {}
    to_log = {
        "Total_Bilirubinlog": total_bilirubin,
        "Alkaline_Phosphotaselog": alkaline_phosphotase,
        "Alamine_Aminotransferaselog": almaine_aminotransferase,
        "Aspartate_Aminotransferaselog": aspartate_aminotransferase
    }
    for i in to_log:
        log_values[i] = np.log1p(to_log[i])
    
    # Input Data
    input_data = {
        "Age": age,
        "Gender": gender_encoded,
        "Total_Bilirubinlog": log_values["Total_Bilirubinlog"],
        "Alkaline_Phosphotaselog": log_values["Alkaline_Phosphotaselog"],
        "Alamine_Aminotransferaselog": log_values["Alamine_Aminotransferaselog"],
        "Aspartate_Aminotransferaselog": log_values["Aspartate_Aminotransferaselog"],
        "Total_Protiens": total_protiens,
        "Albumin": albumin,
        "Albumin_and_Globulin_Ratio": albumin_and_globulin_ratio
    }
    return input_data

# Kidney Disease Input Form
def kidney_form():
    st.subheader("Chronic Kidney Disease - Patient Details")
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 4.0, 90.0, 25.0)
        bp = st.number_input("Blood Pressure", 50.0, 180.0, 110.0)
        sg = st.number_input("Specific Gravity", 1.005, 1.025, 1.006)
        al = st.number_input("Albumin",0, 5, 1)
        su = st.number_input("Sugar", 0, 5, 2)
        rbc = st.selectbox("Red Blood Cells", ("Normal", "Abnormal"))
        pc = st.selectbox("Pus Cell", ("Normal", "Abnormal"))
        pcc = st.selectbox("Pus Cells Clumps", ("Present", "Not Present"))
    with col2:
        ba = st.selectbox("Bacteria", ("Present", "Not Present"))
        bgr = st.number_input("Blood Glucose Random", 20.0, 500.0, 200.0)
        bu = st.number_input("Blood Urea", 1.0, 400.0, 200.0)
        sc = st.number_input("Serum Creatinine", 0.0, 5.0, 2.0)
        sod = st.number_input("Sodium", 4.0, 200.0, 100.0)
        pot = st.number_input("Potassium", 2.0, 50.0, 20.0)
        hemo = st.number_input("Hemoglobin", 3.0, 20.0, 10.0)
        rc = st.number_input("Red Blood Cells Count", 1.0, 10.0, 3.0)
    with col3:
        wc = st.number_input("White Blood Cell Count", 1000.0, 30000.0, 18000.0)
        htn = st.selectbox("Hypertension", ("Yes", "No"))
        dm = st.selectbox("Diabetes Mellitus", ("Yes", "No"))
        cad = st.selectbox("Coronary Artery Disease", ("Yes", "No"))
        appet = st.selectbox("Appetite", ("Good", "Poor"))
        pe = st.selectbox("Pedal Edema", ("Yes", "No"))
        ane = st.selectbox("Anemia", ("Yes", "No"))

    # Encode categorical input values
    values = {
        "Normal": 1,
        "Abnormal": 0,
        "Present": 1,
        "Not Present": 0,
        "Yes": 1,
        "No": 0,
        "Good": 1,
        "Poor": 0
    }
    rbc_encoded = values[rbc]
    pc_encoded = values[pc]
    pcc_encoded = values[pcc]
    ba_encoded = values[ba]
    htn_encoded = values[htn]
    dm_encoded = values[dm]
    cad_encoded = values[cad]
    appet_encoded = values[appet]
    pe_encoded = values[pe]
    ane_encoded = values[ane]

    # Log Transform
    log_values = {}
    to_log = {
        "bp_log": bp,
        "bgr_log": bgr,
        "bu_log": bu,
        "sc_log": sc,
        "pot_log": pot,
        "wc_log": wc
    }
    for i in to_log:
        log_values[i] = np.log1p(to_log[i])

    input_data = {
        "age": age,
        "bp_log": log_values["bp_log"],
        "sg": sg,
        "al": al,
        "su": su,
        "rbc": rbc_encoded,
        "pc": pc_encoded,
        "pcc": pcc_encoded,
        "ba": ba_encoded,
        "bgr_log": log_values["bgr_log"],
        "bu_log": log_values["bu_log"],
        "sc_log": log_values["sc_log"],
        "sod": sod,
        "pot_log": log_values["pot_log"],
        "hemo": hemo,
        "rc": rc,
        "wc_log": log_values["wc_log"],
        "htn": htn_encoded,
        "dm": dm_encoded,
        "cad": cad_encoded,
        "appet": appet_encoded,
        "pe": pe_encoded,
        "ane": ane_encoded,
    }
    return input_data


# Predict Disease
def predict_disease(disease, model_key, input_data, files):
    # Features
    disease = disease.lower()
    feature_names = files[disease+"_features"]
    # Scaler
    scaler = files[disease+"_scaler"]
    # Model
    model = files[disease+"_"+model_key]
    # make dataframe
    df = pd.DataFrame([input_data])
    df = df[feature_names]
    # Scaling
    df_scaled = scaler.transform(df)
    # Prediction probability
    pred_prob = model.predict_proba(df_scaled)[0][1]
    # Liver Logistic Regression uses custom threshold 0.45, others use default 0.5
    if disease == "liver" and model_key == "lr":
        pred = int(pred_prob >= 0.45)
    else:
        pred = model.predict(df_scaled)[0]
    return pred, pred_prob

# Show results
def show_result(disease, pred, pred_prob):
    prob_pct = pred_prob * 100
    st.divider()
    if pred == 1:
        st.error(f"Patient might have {disease}.")
    else:
        st.success(f"Patient might not have {disease}.")
    st.caption(f"Predicted probability: {prob_pct:.1f}%")
    st.progress(int(prob_pct))

# Mode Selection
with st.sidebar:
    st.header("Options")
    mode = st.radio("Choose Mode", ("Basic", "Advanced"))

# Basic Mode
if mode == "Basic":
    st.header("Basic Mode")
    disease_select = st.selectbox("Select Disease", ("Parkinsons", "Liver", "Kidney"), width=250)
    # Disease Form
    if disease_select == "Parkinsons":
        st.text("Find the possibility of having/not having Parkinsons. Predict by filling the details below.")
        input_data = parkinsons_form()
    elif disease_select == "Liver":
        st.text("Find the possibility of having/not having Liver Disease. Predict by filling the details below.")
        input_data = liver_form()
    else:
        st.text("Find the possibility of having/not having Chronic Kidney Disease. Predict by filling the details below.")
        input_data = kidney_form()
    # Button
    if st.button("Predict"):
        best_model_key = {"Parkinsons": "xg", "Liver": "lr", "Kidney": "rf"}[disease_select]
        pred, pred_prob = predict_disease(disease_select, best_model_key, input_data, files)
        show_result(disease_select, pred, pred_prob)

# Advanced Mode
else:
    st.header("Advanced Mode")
    col1, col2 = st.columns(2)
    with col1:
        model_select = st.selectbox("Select Model", ("Logistic Regression", "Random Forest", "XGBoost"))
    with col2:
        disease_select = st.selectbox("Select Disease", ("Parkinsons", "Liver", "Kidney"))

    if model_select == "Logistic Regression":
        if disease_select == "Parkinsons":
            st.text("Find the possibility of having/not having Parkinsons. Predict by filling the details below.")
            input_data = parkinsons_form()
        elif disease_select == "Liver":
            st.text("Find the possibility of having/not having Liver. Predict by filling the details below.")
            input_data = liver_form()
        else:
            st.text("Find the possibility of having/not having Kidney. Predict by filling the details below.")
            input_data = kidney_form()
        if st.button("Predict"):
            pred, pred_prob = predict_disease(disease_select, "lr", input_data, files)
            show_result(disease_select, pred, pred_prob)
    
    elif model_select == "Random Forest":
        if disease_select == "Parkinsons":
            input_data = parkinsons_form()
        elif disease_select == "Liver":
            input_data = liver_form()
        else:
            input_data = kidney_form()
        if st.button("Predict"):
            pred, pred_prob = predict_disease(disease_select, "rf", input_data, files)
            show_result(disease_select, pred, pred_prob)

    else:
        if disease_select == "Parkinsons":
            input_data = parkinsons_form()
        elif disease_select == "Liver":
            input_data = liver_form()
        else:
            input_data = kidney_form()
        if st.button("Predict"):
            pred, pred_prob = predict_disease(disease_select, "xg", input_data, files)
            show_result(disease_select, pred, pred_prob)