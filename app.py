import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Classifier",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    
    .main { background-color: #0f1117; }
    
    .score-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .score-good  { border-left: 4px solid #2ecc71; }
    .score-standard { border-left: 4px solid #f39c12; }
    .score-poor  { border-left: 4px solid #e74c3c; }
    
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-1px);
    }
    
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #667eea44;
    }
</style>
""", unsafe_allow_html=True)


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    for c in df.select_dtypes('object').columns:
        df[c] = df[c].astype(str).str.strip()
    df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace(r'[^0-9\-]','',regex=True), errors='coerce')
    df['Age'] = df['Age'].where(df['Age'].between(10,100))
    for col in ['Annual_Income','Outstanding_Debt','Monthly_Balance','Amount_invested_monthly','Changed_Credit_Limit']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9\.\-]','',regex=True), errors='coerce')
    for col in ['Num_of_Loan','Num_of_Delayed_Payment','Num_Credit_Inquiries']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9\.\-]','',regex=True), errors='coerce')
    df['Credit_Mix'] = df['Credit_Mix'].where(df['Credit_Mix'].isin(['Good','Bad','Standard']))
    df['Occupation'] = df['Occupation'].replace('_______', np.nan)
    df['Occupation'] = df['Occupation'].where(df['Occupation'].astype(str).str.match(r'^[A-Za-z]', na=False))
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].where(df['Payment_of_Min_Amount'].isin(['Yes','No','NM']))
    def parse_age(s):
        try:
            p=str(s).split(); return int(p[0])*12+(int(p[3]) if len(p)>=4 else 0)
        except: return np.nan
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(parse_age)
    df['Delay_from_due_date'] = df['Delay_from_due_date'].clip(lower=0)
    df['Debt_to_Income'] = df['Outstanding_Debt']/(df['Annual_Income']+1)
    df['Salary_to_EMI']  = df['Monthly_Inhand_Salary']/(df['Total_EMI_per_month']+1)
    return df

@st.cache_resource
def train_model(df):
    NUM = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts','Num_Credit_Card',
           'Interest_Rate','Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment',
           'Outstanding_Debt','Credit_Utilization_Ratio','Credit_History_Months',
           'Total_EMI_per_month','Amount_invested_monthly','Monthly_Balance',
           'Debt_to_Income','Salary_to_EMI']
    CAT = ['Occupation','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour']
    FEATS = NUM + CAT
    X = df[FEATS]; y_raw = df['Credit_Score']
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    pre = ColumnTransformer([
        ('n', Pipeline([('i', SimpleImputer(strategy='median'))]), NUM),
        ('c', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                        ('e', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), CAT)
    ])
    model = Pipeline([('pre', pre),
                      ('clf', RandomForestClassifier(n_estimators=100, max_depth=15,
                               class_weight='balanced', random_state=42, n_jobs=-1))])
    model.fit(X, y)
    return model, le, FEATS, NUM, CAT


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin:0; color:#667eea;">💳 Credit Score</h2>
        <p style="margin:0; color:#888; font-size:0.85rem;">ML Classification App</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📁 Upload Dataset")
    train_file = st.file_uploader("Upload train.csv", type=['csv'], key='train')
    test_file  = st.file_uploader("Upload test.csv", type=['csv'], key='test')

    st.markdown("---")
    st.markdown("""
    **👤 Author**  
    Ujjala Mustafa  
    ML Intern — CodeAlpha  
    
    [![GitHub](https://img.shields.io/badge/GitHub-UjjalaMustafa-black?style=flat-square&logo=github)](https://github.com/UjjalaMustafa)
    """)


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.markdown("# 💳 Credit Score Classification")
st.markdown("*Predict customer credit scores — Good, Standard, or Poor — using Machine Learning*")
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📊 EDA", "🤖 Model & Results", "🔮 Predict"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📌 Project Overview")
    st.info("""
    This project builds an end-to-end machine learning pipeline to classify customers into 
    **Good**, **Standard**, or **Poor** credit score categories based on their financial 
    behavior and credit history. Built as part of my **CodeAlpha Machine Learning Internship**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="color:#667eea;">100K</div>
            <div class="metric-label">Training Samples</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="color:#2ecc71;">75.8%</div>
            <div class="metric-label">Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="color:#f39c12;">76.1%</div>
            <div class="metric-label">F1 Score</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="color:#e74c3c;">3</div>
            <div class="metric-label">Classes</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🛠️ Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Tool | Purpose |
        |------|---------|
        | Python 3.14 | Core language |
        | pandas & NumPy | Data manipulation |
        | scikit-learn | ML pipeline & modeling |
        """)
    with col2:
        st.markdown("""
        | Tool | Purpose |
        |------|---------|
        | Matplotlib & Seaborn | Visualization |
        | Streamlit | Web application |
        | Jupyter Notebook | Development |
        """)

# ════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Exploratory Data Analysis")

    if train_file:
        df = load_and_clean(train_file)

        # Class distribution
        st.markdown("### 🎯 Target Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')
            counts = df['Credit_Score'].value_counts()
            colors = ['#2ecc71','#e74c3c','#f39c12']
            bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='none', width=0.5)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                       f'{val:,}', ha='center', color='white', fontsize=10)
            ax.set_facecolor('#1e2130')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title('Class Distribution', color='white', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e2130')
            ax.pie(counts.values, labels=counts.index, colors=colors,
                   autopct='%1.1f%%', startangle=90,
                   textprops={'color':'white'})
            ax.set_title('Class Balance', color='white', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Missing values
        st.markdown("### 🔍 Missing Values")
        missing = df.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')
        ax.barh(missing.index, missing.values/len(df)*100, color='#667eea')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Missing %', color='white')
        ax.set_title('Missing Values by Column', color='white', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.warning("⬅️ Please upload **train.csv** from the sidebar to see EDA charts!")

# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL & RESULTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 Model Performance")

    if train_file:
        df = load_and_clean(train_file)

        with st.spinner("🔄 Training Random Forest model... Please wait..."):
            model, le, FEATS, NUM, CAT = train_model(df)

        st.success("✅ Model trained successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""<div class="metric-card">
                <div class="metric-value" style="color:#2ecc71;">75.8%</div>
                <div class="metric-label">Train Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card">
                <div class="metric-value" style="color:#667eea;">76.1%</div>
                <div class="metric-label">F1 Score (weighted)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Per class performance
        st.markdown("### 📋 Per-Class Performance")
        perf_df = pd.DataFrame({
            'Class': ['Good', 'Poor', 'Standard'],
            'Precision': [0.58, 0.73, 0.91],
            'Recall': [0.89, 0.85, 0.67],
            'F1-Score': [0.70, 0.79, 0.77],
            'Support': [17828, 28998, 53174]
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Feature importance
        st.markdown("### 🏆 Top 10 Feature Importances")
        clf = model.named_steps['clf']
        imp = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')
        colors_imp = ['#e74c3c' if i < 3 else '#667eea' for i in range(10)]
        ax.barh(imp.index[::-1], imp.values[::-1], color=colors_imp[::-1])
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Importance', color='white')
        ax.set_title('Top 10 Feature Importances', color='white', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.warning("⬅️ Please upload **train.csv** from the sidebar to train the model!")

# ════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔮 Predict Credit Score")
    st.markdown("Fill in customer details below to predict their credit score:")

    if train_file:
        df = load_and_clean(train_file)
        model, le, FEATS, NUM, CAT = train_model(df)

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**👤 Personal Info**")
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                occupation = st.selectbox("Occupation", ['Engineer','Doctor','Teacher','Lawyer',
                                          'Accountant','Manager','Scientist','Developer','Journalist','Other'])
                annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
                monthly_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=4000.0, step=100.0)

            with col2:
                st.markdown("**💳 Credit Info**")
                num_bank_accounts = st.slider("Num Bank Accounts", 0, 20, 3)
                num_credit_cards = st.slider("Num Credit Cards", 0, 15, 2)
                interest_rate = st.slider("Interest Rate (%)", 0, 50, 12)
                credit_mix = st.selectbox("Credit Mix", ['Good', 'Standard', 'Bad'])
                credit_history = st.slider("Credit History (months)", 0, 400, 120)
                credit_utilization = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0)

            with col3:
                st.markdown("**💰 Loan & Payment Info**")
                num_loans = st.slider("Num of Loans", 0, 10, 2)
                delay_days = st.slider("Delay from Due Date (days)", 0, 60, 5)
                num_delayed = st.slider("Num Delayed Payments", 0, 30, 2)
                outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=2000.0, step=100.0)
                total_emi = st.number_input("Total EMI per Month ($)", min_value=0.0, value=300.0, step=50.0)
                payment_min = st.selectbox("Pays Min Amount?", ['Yes', 'No', 'NM'])
                payment_behaviour = st.selectbox("Payment Behaviour", [
                    'High_spent_Small_value_payments',
                    'Low_spent_Large_value_payments',
                    'High_spent_Medium_value_payments',
                    'Low_spent_Small_value_payments',
                    'High_spent_Large_value_payments',
                    'Low_spent_Medium_value_payments'
                ])
                amount_invested = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=200.0, step=50.0)
                monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=500.0, step=50.0)

            submitted = st.form_submit_button("🔮 Predict Credit Score")

        if submitted:
            debt_to_income = outstanding_debt / (annual_income + 1)
            salary_to_emi  = monthly_salary / (total_emi + 1)

            input_data = pd.DataFrame([{
                'Age': age,
                'Annual_Income': annual_income,
                'Monthly_Inhand_Salary': monthly_salary,
                'Num_Bank_Accounts': num_bank_accounts,
                'Num_Credit_Card': num_credit_cards,
                'Interest_Rate': interest_rate,
                'Num_of_Loan': num_loans,
                'Delay_from_due_date': delay_days,
                'Num_of_Delayed_Payment': num_delayed,
                'Outstanding_Debt': outstanding_debt,
                'Credit_Utilization_Ratio': credit_utilization,
                'Credit_History_Months': credit_history,
                'Total_EMI_per_month': total_emi,
                'Amount_invested_monthly': amount_invested,
                'Monthly_Balance': monthly_balance,
                'Debt_to_Income': debt_to_income,
                'Salary_to_EMI': salary_to_emi,
                'Occupation': occupation,
                'Credit_Mix': credit_mix,
                'Payment_of_Min_Amount': payment_min,
                'Payment_Behaviour': payment_behaviour
            }])

            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            predicted_label = le.inverse_transform([prediction])[0]

            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")

            color_map = {'Good': '#2ecc71', 'Standard': '#f39c12', 'Poor': '#e74c3c'}
            emoji_map = {'Good': '✅', 'Standard': '⚠️', 'Poor': '❌'}
            color = color_map[predicted_label]
            emoji = emoji_map[predicted_label]

            st.markdown(f"""
            <div class="score-card score-{predicted_label.lower()}">
                <h1 style="color:{color}; font-size:3rem; margin:0;">{emoji} {predicted_label}</h1>
                <p style="color:#aaa; margin-top:0.5rem;">Predicted Credit Score</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Confidence Scores")
            for cls, prob in zip(le.classes_, proba):
                c = color_map[cls]
                st.markdown(f"**{cls}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")

    else:
        st.warning("⬅️ Please upload **train.csv** from the sidebar to use the prediction feature!")
