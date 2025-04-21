import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- DATA ----------------------
accuracy_scores = {
    'Logistic Regression': 92.04,
    'SVM': 94.66,
    'Decision Tree': 96.29,
    'KNN': 93.67,
    'Random Forest': 97.06
}

# Updated feature importance data (sorted in ascending order)
feature_importance = {
    'SSLfinal_State': 0.312944,
    'URL_of_Anchor': 0.267490,
    'having_Sub_Domain': 0.084409,
    'web_traffic': 0.083660,
    'Links_in_tags': 0.050256,
    'Prefix_Suffix': 0.044323,
    'Request_URL': 0.022186,
    'Links_pointing_to_page': 0.021330,
    'age_of_domain': 0.016779,
    'Domain_registeration_length': 0.016453,
    'having_IP_Address': 0.014493,
    'DNSRecord': 0.014174,
    'Google_Index': 0.013711,
    'URL_Length': 0.011269,
    'HTTPS_token': 0.007473,
    'on_mouseover': 0.007346,
    'Statistical_report': 0.006077,
    'Abnormal_URL': 0.005628
}

# Confusion matrix static data
conf_matrices = {
    'Logistic Regression': [[860, 96], [80, 1175]],
    'SVM': [[882, 74], [44, 1211]],
    'Decision Tree': [[917, 39], [43, 1212]],
    'KNN': [[882, 74], [66, 1189]],
    'Random Forest': [[917, 39], [26, 1229]]
}

# Sort the feature importance in ascending order
sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1]))

# ---------------------- PAGE SETTINGS ----------------------
st.set_page_config(page_title="Phishing Detection Dashboard", layout="wide")

# ---------------------- HEADER ----------------------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🔐 Phishing Website Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------- COLUMNS ----------------------
col1, col2 = st.columns([2, 1])

# ---------- LEFT: Accuracy Chart ----------
with col1:
    st.subheader("📊 Model Accuracy Comparison")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(accuracy_scores.keys(), accuracy_scores.values(), color='#5DADE2', edgecolor='black')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(90, 100)
    ax1.set_title("Model-wise Accuracy")
    for i, (model, acc) in enumerate(accuracy_scores.items()):
        ax1.text(i, acc + 0.2, f"{acc}%", ha='center', fontsize=10)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# ---------- RIGHT: Model Info Box ----------
with col2:
    st.subheader("🎯 Selected Model Info")
    selected_model = st.selectbox("Choose a model:", list(accuracy_scores.keys()))
    st.metric(label="Accuracy", value=f"{accuracy_scores[selected_model]}%")

    # Confusion Matrix Display
    st.markdown("### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrices[selected_model], annot=True, fmt='d', cmap='coolwarm', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix - {selected_model}")
    st.pyplot(fig_cm)

# ---------------------- FEATURE IMPORTANCE ----------------------
st.subheader("🔍 Top Feature Importances")
st.markdown("The following features are the most significant in determining whether a website is phishing or not.")

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x=list(sorted_feature_importance.values()), y=list(sorted_feature_importance.keys()), ax=ax2, palette='crest')
ax2.set_title("Top Contributing Features")
ax2.set_xlabel("Importance Score")
st.pyplot(fig2)

# ---------------------- FOOTER ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Made by Jatin | L.D. College of Engineering</div>", unsafe_allow_html=True)
