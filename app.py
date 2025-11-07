
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance
import pickle, io, base64, time

st.set_page_config(layout="wide", page_title="HR Attrition Dashboard")

@st.cache_data
def load_sample_data(path=None, n=200):
    if path is not None:
        return pd.read_csv(path)
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "Age": rng.randint(22,60,size=n),
        "BusinessTravel": rng.choice(["Travel_Rarely","Travel_Frequently","Non-Travel"], size=n),
        "DailyRate": rng.randint(200,1500,size=n),
        "Department": rng.choice(["Sales","Research & Development","HR"], size=n),
        "DistanceFromHome": rng.randint(1,30,size=n),
        "Education": rng.randint(1,5,size=n),
        "JobRole": rng.choice(["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager"], size=n),
        "JobSatisfaction": rng.choice([1,2,3,4], size=n, p=[0.2,0.3,0.3,0.2]),
        "OverTime": rng.choice(["Yes","No"], size=n, p=[0.2,0.8]),
        "MonthlyIncome": rng.randint(1000,20000,size=n),
        "NumCompaniesWorked": rng.randint(0,7,size=n),
        "TotalWorkingYears": rng.randint(1,40,size=n),
        "YearsAtCompany": rng.randint(0,20,size=n),
        "Attrition": rng.choice(["Yes","No"], size=n, p=[0.18,0.82])
    })
    return df

def detect_satisfaction_col(df):
    for c in df.columns:
        if 'satisf' in c.lower():
            return c
    return None

def detect_jobrole_col(df):
    for c in df.columns:
        if 'job' in c.lower() and 'role' in c.lower():
            return c
    for c in df.columns:
        if 'role' in c.lower():
            return c
    for c in df.columns:
        if c.lower() in ['jobrole','role','position']:
            return c
    return None

def build_preprocessor(df, numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer([('num', num_pipeline, numeric_features), ('cat', cat_pipeline, categorical_features)], remainder='drop', verbose_feature_names_out=False)
    return preprocessor, numeric_features, categorical_features

def preprocess_fit_transform(preprocessor, X_train, X_test):
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder' and trans == 'drop':
            continue
        if hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
            ohe = trans.named_steps['onehot']
            for i, c in enumerate(cols):
                cats = ohe.categories_[i]
                feature_names.extend([f"{c}_{str(cat)}" for cat in cats])
        else:
            feature_names.extend(list(cols))
    if len(feature_names) != X_train_t.shape[1]:
        feature_names = [f"f_{i}" for i in range(X_train_t.shape[1])]
    return X_train_t, X_test_t, feature_names

def compute_model_metrics(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    proba = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None
    metrics = {
        "train_acc": accuracy_score(y_train, y_train_pred),
        "test_acc": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "auc": roc_auc_score(y_test, proba) if proba is not None else None,
        "y_test_proba": proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    }
    return metrics, clf

st.title("HR Attrition — Interactive Dashboard & Modeling")

uploaded = st.file_uploader("Upload CSV dataset", type=['csv'], key="uploader")
use_sample = st.checkbox("Use sample dataset instead", value=True)
if uploaded is not None:
    df = load_sample_data(path=uploaded)
elif use_sample:
    df = load_sample_data(path=None, n=500)
else:
    st.info("Please upload a dataset or enable sample dataset to proceed.")
    st.stop()

target_col = None
candidates = [c for c in df.columns if any(k in c.lower() for k in ['attrition','left','churn','exited','target','label','leave','resign'])]
if len(candidates) == 1:
    target_col = candidates[0]
elif len(candidates) > 1:
    target_col = st.selectbox("Multiple candidate target columns detected. Choose target column:", candidates)
else:
    bin_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]
    if len(bin_cols) == 1:
        target_col = bin_cols[0]
    else:
        st.write("Detected columns:")
        st.write(df.columns.tolist())
        target_col = st.selectbox("Select the target column from the list above:", df.columns.tolist())

y_raw = df[target_col]
def map_binary(y_raw):
    mapped = None
    if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
        vals = y_raw.dropna().unique().tolist()
        if len(vals) == 2:
            m = {}
            v0 = str(vals[0]).strip().lower()
            v1 = str(vals[1]).strip().lower()
            yes_vals = {'yes','y','true','1','t','left','churn','exited'}
            if v0 in yes_vals:
                m[vals[0]] = 1; m[vals[1]] = 0
            elif v1 in yes_vals:
                m[vals[1]] = 1; m[vals[0]] = 0
            else:
                m[vals[0]] = 0; m[vals[1]] = 1
            mapped = y_raw.map(m)
    else:
        if set(y_raw.dropna().unique()).issubset({0,1}):
            mapped = y_raw
    return mapped

y = map_binary(y_raw)
if y is None:
    st.warning("Target could not be auto-mapped to binary. Modeling disabled.")
    modeling_allowed = False
else:
    modeling_allowed = True
    df['_target_binary_'] = y

job_col = detect_jobrole_col(df)
satisf_col = detect_satisfaction_col(df)
st.sidebar.header("Global Filters")
if job_col is not None:
    roles = sorted(df[job_col].dropna().unique().tolist())
    selected_roles = st.sidebar.multiselect("Filter by Job Role", options=roles, default=roles)
else:
    selected_roles = None

if satisf_col is not None:
    min_s, max_s = int(df[satisf_col].min()), int(df[satisf_col].max())
    s_range = st.sidebar.slider(f"Filter by {satisf_col}", min_value=min_s, max_value=max_s, value=(min_s, max_s))
else:
    s_range = None

df_filtered = df.copy()
if job_col is not None and selected_roles is not None and len(selected_roles)>0:
    df_filtered = df_filtered[df_filtered[job_col].isin(selected_roles)]
if satisf_col is not None and s_range is not None:
    df_filtered = df_filtered[(df_filtered[satisf_col] >= s_range[0]) & (df_filtered[satisf_col] <= s_range[1])]

tab1, tab2, tab3 = st.tabs(["Dashboard", "Train & Evaluate Models", "Predict New Data"])

with tab1:
    st.header("HR Dashboard — Charts & Insights")
    st.markdown("Use the sidebar filters (Job Role & Satisfaction) to apply to all charts.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("1) Attrition Rate by Job Role")
        if job_col is not None:
            pivot = df_filtered.groupby(job_col)['_target_binary_'].mean().reset_index().rename(columns={'_target_binary_':'attrition_rate'})
            fig = px.bar(pivot, x=job_col, y='attrition_rate', color='attrition_rate', labels={'attrition_rate':'Attrition Rate'}, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No job role column detected.")

        st.subheader("2) Satisfaction Distribution by Attrition (Boxplot)")
        if satisf_col is not None:
            fig2 = px.box(df_filtered, x='_target_binary_', y=satisf_col, labels={'_target_binary_':'Attrition (0=No,1=Yes)'})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No satisfaction-like column detected.")

        st.subheader("3) Tenure (YearsAtCompany) Distribution by Attrition")
        if 'YearsAtCompany' in df.columns:
            fig3 = px.histogram(df_filtered, x='YearsAtCompany', color='_target_binary_', barmode='overlay', nbins=20)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No YearsAtCompany column detected.")

    with col2:
        st.subheader("4) Attrition Heatmap (JobRole vs OverTime)")
        if job_col is not None and 'OverTime' in df.columns:
            heat = df_filtered.groupby([job_col,'OverTime'])['_target_binary_'].mean().reset_index()
            heat_pivot = heat.pivot(index=job_col, columns='OverTime', values='_target_binary_').fillna(0)
            fig4, ax4 = plt.subplots(figsize=(4,6))
            sns.heatmap(heat_pivot, annot=True, fmt=".2f", cmap='Reds', ax=ax4)
            ax4.set_title("Attrition rate by Role & OverTime")
            st.pyplot(fig4)
        else:
            st.info("Requires JobRole and OverTime columns for heatmap.")

        st.subheader("5) Income vs Satisfaction vs Attrition (Scatter + Trend)")
        if 'MonthlyIncome' in df.columns and satisf_col is not None:
            fig5 = px.scatter(df_filtered, x='MonthlyIncome', y=satisf_col, color='_target_binary_', trendline='ols', labels={satisf_col:'Satisfaction'})
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Requires MonthlyIncome and satisfaction column.")

with tab2:
    st.header("Train & Evaluate Models")
    st.markdown("Choose models to train and click **Run Models**. Models will be trained on the filtered dataset.")
    if not modeling_allowed:
        st.warning("Modeling not available because target could not be mapped to binary. Please upload a dataset with a binary target.")
    else:
        models_to_run = st.multiselect("Select models to run", options=['DecisionTree','RandomForest','GradientBoosting'], default=['DecisionTree','RandomForest','GradientBoosting'])
        run_button = st.button("Run Models")
        if run_button:
            with st.spinner("Training models... this may take a moment"):
                model_df = df_filtered.dropna(subset=['_target_binary_']).copy()
                X = model_df.drop(columns=[target_col,'_target_binary_']) if target_col in model_df.columns else model_df.drop(columns=['_target_binary_'])
                y = model_df['_target_binary_'].astype(int)
                numeric_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
                categorical_feats = X.select_dtypes(include=['object','category','bool']).columns.tolist()
                preprocessor, numeric_feats, categorical_feats = build_preprocessor(X, numeric_feats, categorical_feats)
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                X_train, X_test, feature_names = preprocess_fit_transform(preprocessor, X_train_raw, X_test_raw)

                results = []
                artifacts = {}
                for m in models_to_run:
                    if m == 'DecisionTree':
                        clf = DecisionTreeClassifier(random_state=42)
                    elif m == 'RandomForest':
                        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    elif m == 'GradientBoosting':
                        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    else:
                        continue
                    metrics, trained = compute_model_metrics(clf, X_train, y_train, X_test, y_test)
                    pipe = Pipeline([('preprocessor', preprocessor), ('clf', trained.__class__(**{k: v for k,v in trained.get_params().items() if k!='random_state'}))])
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
                    metrics['cv_mean'] = cv_scores.mean(); metrics['cv_std'] = cv_scores.std()
                    try:
                        fi = trained.feature_importances_
                    except Exception:
                        fi = None
                    metrics['feature_importances'] = fi
                    results.append((m, metrics, trained))
                    artifacts[m] = {'model':trained, 'preprocessor':preprocessor, 'feature_names':feature_names}

                summary_rows = []
                for name, metrics, trained in results:
                    summary_rows.append({
                        'model': name,
                        'train_acc': metrics['train_acc'],
                        'test_acc': metrics['test_acc'],
                        'cv_mean': metrics.get('cv_mean', None),
                        'cv_std': metrics.get('cv_std', None),
                        'precision_macro': metrics['precision_macro'],
                        'recall_macro': metrics['recall_macro'],
                        'f1_macro': metrics['f1_macro'],
                        'auc': metrics.get('auc', None)
                    })
                summary_df = pd.DataFrame(summary_rows)
                st.subheader("Model Summary")
                st.dataframe(summary_df.style.highlight_max(subset=['test_acc'], color='lightgreen'))

                for name, metrics, trained in results:
                    st.markdown(f"### {name}")
                    st.write("**Confusion Matrix**")
                    cm = metrics['confusion_matrix']
                    fig, ax = plt.subplots(figsize=(4,3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                    st.pyplot(fig)
                    st.write("**Classification Report**")
                    cr_df = pd.DataFrame(metrics['classification_report']).transpose()
                    st.dataframe(cr_df)
                    if metrics.get('y_test_proba') is not None and metrics.get('auc') is not None:
                        fpr, tpr, _ = roc_curve(y_test, metrics['y_test_proba'])
                        precision, recall, _ = precision_recall_curve(y_test, metrics['y_test_proba'])
                        ap = average_precision_score(y_test, metrics['y_test_proba'])
                        fig2, ax2 = plt.subplots(1,2, figsize=(10,4))
                        ax2[0].plot(fpr, tpr, label=f"AUC={metrics['auc']:.3f}"); ax2[0].plot([0,1],[0,1],'k--', linewidth=0.8)
                        ax2[0].set_title("ROC Curve"); ax2[0].set_xlabel("FPR"); ax2[0].set_ylabel("TPR")
                        ax2[1].plot(recall, precision, label=f"AP={ap:.3f}"); ax2[1].set_title("Precision-Recall"); ax2[1].set_xlabel("Recall"); ax2[1].set_ylabel("Precision")
                        st.pyplot(fig2)

                st.session_state['artifacts'] = artifacts
                st.session_state['last_results'] = summary_df
                st.success("Models trained and results displayed.")

with tab3:
    st.header("Predict New Data & Download Predictions")
    new_upload = st.file_uploader("Upload new data CSV for prediction", type=['csv'], key="predict_uploader")
    if 'artifacts' not in st.session_state:
        st.info("No trained models found in session. Train models first in 'Train & Evaluate Models' tab.")
    else:
        model_names = list(st.session_state['artifacts'].keys())
        chosen = st.selectbox("Select model to use for prediction", options=model_names)
        if new_upload is not None:
            new_df = pd.read_csv(new_upload)
            st.write("Preview of uploaded data:")
            st.dataframe(new_df.head())
            if st.button("Run Prediction"):
                art = st.session_state['artifacts'][chosen]
                model = art['model']; preprocessor = art['preprocessor']; feature_names = art['feature_names']
                Xnew = new_df.copy()
                possible_targets = [c for c in Xnew.columns if any(k in c.lower() for k in ['attrition','target','label','left','churn'])]
                if len(possible_targets)>0:
                    Xnew = Xnew.drop(columns=possible_targets)
                try:
                    Xnew_t = preprocessor.transform(Xnew)
                except Exception as e:
                    st.error(f"Preprocessor failed: {e}. Ensure uploaded data has similar columns as training.")
                    st.stop()
                preds = model.predict(Xnew_t)
                probs = model.predict_proba(Xnew_t)[:,1] if hasattr(model, "predict_proba") else None
                new_df['predicted_attrition'] = preds
                if probs is not None:
                    new_df['predicted_proba'] = probs
                csv = new_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
                st.success("Predictions completed.")

st.sidebar.markdown("---")
st.sidebar.write("Upload dataset and explore attrition analytics.")
