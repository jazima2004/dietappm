# app.py
import streamlit as st
import numpy as np
import joblib

# ==========================
# Load saved model and maps
# ==========================
kmeans = joblib.load("kmeans_prakriti_model.joblib")
cluster_map = joblib.load("cluster_map.joblib")
activity_map = joblib.load("activity_map.joblib")

# ==========================
# Nutrient calculation function
# ==========================
def compute_nutrient_targets(age, gender, height_cm, weight_kg,
                             activity_level='moderate', goal='maintain',
                             prakriti='Vata', health_conditions=None):
    base = {'carbs': 0.55, 'protein': 0.20, 'fat': 0.25}
    activity_factors = {'sedentary':1.2,'light':1.375,'moderate':1.55,'very':1.725,'extra':1.9}
    af = activity_factors.get(activity_level.lower(), 1.55)
    C = 5 if str(gender).lower().startswith('m') else -161
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + C
    tdee = bmr * af
    if goal=='loss': calorie_target = tdee - 500
    elif goal=='gain': calorie_target = tdee + 500
    else: calorie_target = tdee

    # Prakriti adjustments
    prakriti_factors = {'vata':1.05,'pitta':1.00,'kapha':0.95}
    prakriti_deltas = {
        'vata': {'carbs':-0.10,'protein':+0.05,'fat':+0.05},
        'pitta': {'carbs':+0.05,'protein':0.0,'fat':-0.05},
        'kapha': {'carbs':-0.10,'protein':+0.10,'fat':0.0}
    }
    parts = [p.lower() for p in prakriti.split('-')]
    mults = [prakriti_factors.get(p,1.0) for p in parts]
    avg_mult = sum(mults)/len(mults)
    calorie_target *= avg_mult
    deltas = {'carbs':0.0,'protein':0.0,'fat':0.0}
    for p in parts:
        d = prakriti_deltas.get(p)
        if d:
            deltas['carbs'] += d['carbs']
            deltas['protein'] += d['protein']
            deltas['fat'] += d['fat']
    if parts:
        deltas = {k:v/len(parts) for k,v in deltas.items()}

    if health_conditions:
        conds = [c.lower() for c in health_conditions]
        if 'diabetes' in conds: deltas['carbs']-=0.08; deltas['protein']+=0.05
        if 'hypertension' in conds: deltas['fat']-=0.03
        if 'obesity' in conds: calorie_target*=0.9; deltas['carbs']-=0.05; deltas['protein']+=0.05
        if 'kidney' in ' '.join(conds): deltas['protein']-=0.10
        if 'heart' in ' '.join(conds) or 'cardiac' in ' '.join(conds): deltas['fat']-=0.05

    adjusted = {k: base[k]+deltas[k] for k in base}
    for k in adjusted: 
        if adjusted[k]<0.01: adjusted[k]=0.01
    total = sum(adjusted.values())
    adjusted = {k:v/total for k,v in adjusted.items()}

    calories = round(calorie_target)
    carbs_g = round(adjusted['carbs']*calorie_target/4)
    protein_g = round(adjusted['protein']*calorie_target/4)
    fat_g = round(adjusted['fat']*calorie_target/9)

    return {'calories':calories,'carbs_g':carbs_g,'protein_g':protein_g,'fat_g':fat_g}

# ==========================
# Streamlit UI
# ==========================
st.title("Ayurvedic Diet Management System")
st.write("Enter patient details to predict Prakriti and nutrient targets:")

with st.form("patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male","Female"])
    height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg)", min_value=10, max_value=200, value=65)
    activity = st.selectbox("Activity Level", ["sedentary","light","moderate","very","extra"])
    goal = st.selectbox("Goal", ["maintain","gain","loss"])
    health_conditions_str = st.text_input("Health Conditions (comma-separated, e.g., Diabetes,Hypertension)", "")
    submitted = st.form_submit_button("Submit")

    if submitted:
        health_conditions = [x.strip() for x in health_conditions_str.split(",") if x.strip()]
        bmi = weight_kg / (height_cm/100)**2
        activity_num = activity_map.get(activity, 3)
        health_count = len(health_conditions)

        X_new = np.array([[bmi, weight_kg, height_cm, activity_num, health_count]])
        cluster_label = kmeans.predict(X_new)[0]
        prakriti_pred = cluster_map[cluster_label]

        nutrient_targets = compute_nutrient_targets(
            age, gender, height_cm, weight_kg,
            activity_level=activity, goal=goal,
            prakriti=prakriti_pred, health_conditions=health_conditions
        )

        st.subheader("Predicted Prakriti:")
        st.write(prakriti_pred)
        st.subheader("Nutrient Targets:")
        st.write(nutrient_targets)
