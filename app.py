from flask import Flask, request, jsonify, url_for, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import time
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from sklearn.linear_model import BayesianRidge
from scipy.stats import linregress
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Any, List, Dict, Optional

load_dotenv()
warnings.simplefilter("ignore", category=RuntimeWarning)

app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "https://gsci-frontend.onrender.com"]}}, supports_credentials=True)

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
def get_database_connection() -> Any:
    """Create database connection using environment variable"""
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        return create_engine(database_url)
    else:
        raise Exception("DATABASE_URL not found. Please set it in your environment.")

def load_data_from_database() -> pd.DataFrame:
    """Load data from PostgreSQL database"""
    engine = get_database_connection()
    try:
        query = "SELECT * FROM final_cookie_sales_all_years"
        df = pd.read_sql(query, engine)
        print(f"✅ Successfully loaded {len(df)} rows from database")
        return df
    except Exception as e:
        print(f"❌ Error loading from database: {e}")
        raise Exception("Could not load data from database.")

# -------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------
df = load_data_from_database()

if df is None:
    raise Exception("Could not load data from database.")

# Remove the backwards renaming - use new column names directly
# df = df.rename(columns={
#     'SU_Name': 'SU Name',
#     'SU_Num': 'SU #'
# })

# DO NOT force troop_id to int, always treat as string
# df['troop_id'] = df['troop_id'].astype(int)
# Ensure troop_id is formatted as 5-character string with leading zeros for numerical values
df['troop_id'] = df['troop_id'].astype(str).str.strip().apply(lambda x: f"{int(x):05d}" if x.strip().isdigit() else f"{x:>5}")
df['period'] = df['period'].astype(int)
df['number_of_girls'] = df['number_of_girls'].astype(float)
df['number_cases_sold'] = df['number_cases_sold'].astype(float)
df['period_squared'] = df['period'] ** 2

# SU_Num cleaning is now handled exclusively in the ETL pipeline.
# This check will warn if any legacy data with a non-numeric SU_Num is present.
# Ensure SU_Num is string type before applying string operations
df['SU_Num'] = df['SU_Num'].astype(str)
if (df['SU_Num'].str.match(r'^SU', case=False).any() or df['SU_Num'].str.contains(r'[^0-9]', regex=True).any()):
    print("[WARNING] Some SU_Num values in the loaded data are not purely numeric. Please reprocess your data with the updated ETL pipeline.")

# Debug: Check SU_Num values in the loaded data
print(f"[DEBUG] SU_Num values in loaded data: {df['SU_Num'].unique()[:10]}")
print(f"[DEBUG] Years in loaded data: {sorted(df['period'].unique())}")

# Clean SU_Num values if they still have SU prefix (temporary fix)
df['SU_Num'] = df['SU_Num'].astype(str).str.replace(r'^SU\s*', '', regex=True).str.replace(r'^SU', '', regex=True).str.strip()
print(f"[DEBUG] SU_Num values after cleaning: {df['SU_Num'].unique()[:10]}")

# Normalize cookie types
normalized_to_canonical = {
    'adventurefuls': 'Adventurefuls',
    'dosidos': 'Do-Si-Dos',
    'samoas': 'Samoas',
    'smores': "S'mores",
    'tagalongs': 'Tagalongs',
    'thinmints': 'Thin Mints',
    'toffeetastic': 'Toffee-tastic',
    'trefoils': 'Trefoils',
    'lemonups': 'Lemon-Ups'
}

def normalize_cookie_type(raw_name: str) -> str:
    raw_lower = raw_name.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '', raw_lower)
    return normalized_to_canonical.get(slug, raw_name)

df['canonical_cookie_type'] = df['cookie_type'].apply(normalize_cookie_type)

# Add historical stats for interval clamping
stats = df.groupby(['troop_id', 'canonical_cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
stats.columns = ['troop_id', 'canonical_cookie_type', 'historical_low', 'historical_high']
df = df.merge(stats, on=['troop_id', 'canonical_cookie_type'], how='left')

# -------------------------------
# TRAIN RIDGE TO GET RMSE FOR INTERVAL WIDTH
# -------------------------------
def run_ridge_interval_analysis() -> None:
    groups = df.groupby(['troop_id', 'canonical_cookie_type'])
    y_train_all, y_pred_all = [], []

    for (troop, cookie), group in tqdm(groups):
        group = group.sort_values('period')
        train = group[group['period'] <= 4]
        test = group[group['period'] == 5]
        if train.empty or test.empty:
            continue

        X_train = train[['period', 'number_of_girls']]
        y_train = train['number_cases_sold']
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        best_model = Ridge(alpha=1.0)
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_train_scaled)

        y_train_all.extend(y_train)
        y_pred_all.extend(y_pred)

    # Check if we have any data before calculating RMSE
    if len(y_train_all) == 0:
        print("Warning: No valid training data found for RMSE calculation. Using default RMSE of 10.0")
        rmse = 10.0
    else:
        rmse = np.sqrt(mean_squared_error(y_train_all, y_pred_all))
    
    app.config['OVERALL_RIDGE_RMSE'] = rmse
    print(f"Global RMSE for prediction interval: {rmse:.2f}")

run_ridge_interval_analysis()

# -------------------------------
# API ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template("home.html")

@app.route('/healthz')
def health_check():
    # Check if we're using database or CSV
    data_source = "database" if get_database_connection() else "csv"
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "data_source": data_source,
        "data_rows": len(df) if df is not None else 0
    }), 200

@app.route('/predict')
def predict_page():
    return render_template("index.html")

@app.route('/api/troop_ids')
def get_troop_ids():
    # Get all unique troop IDs and format them as 5-character strings
    df_clean = df.copy()
    df_clean['troop_id'] = df_clean['troop_id'].astype(str).str.strip()
    
    # Get unique troop IDs and format them as 5-character strings with leading zeros for numerical values
    unique_troop_ids = df_clean['troop_id'].unique().tolist()
    formatted_troop_ids = []
    for troop_id in unique_troop_ids:
        troop_id_clean = troop_id.strip()
        if troop_id_clean not in ['SU', 'nan', '']:
            if troop_id_clean.isdigit():
                formatted_troop_ids.append(f"{int(troop_id_clean):05d}")
            else:
                formatted_troop_ids.append(f"{troop_id_clean:>5}")
    
    # Return sorted, formatted troop IDs
    return jsonify(sorted(formatted_troop_ids))

# In /api/predict, remove fallback to CSV as well
# Replace the data loading logic in api_predict to use only the database
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Always load the active_cookies table and build the image map at the start
    engine = get_database_connection()
    active_df = pd.read_sql("SELECT * FROM active_cookies", engine)
    # Normalize cookie_type in active_df for mapping
    active_df['normalized_cookie_type'] = active_df['cookie_type'].apply(normalize_cookie_type)
    cookie_image_map = dict(zip(active_df['normalized_cookie_type'], active_df['image_filename']))
    print("[DEBUG] cookie_image_map:", cookie_image_map)
    try:
        # Get request parameters: troop_id and num_girls.
        req_data = request.get_json() or {}
        troop_id = str(req_data.get("troop_id", "")).strip()
        input_num_girls = float(req_data.get("num_girls", 0))
        if not troop_id or input_num_girls <= 0:
            print(f"[DEBUG] Invalid troop_id ({troop_id}) or num_girls ({input_num_girls})")
            return jsonify({"error": "Invalid troop_id or num_girls"}), 400

        # Load data from database only
        df_new = load_data_from_database()
        if df_new is None:
            print("[DEBUG] Could not load data from database")
            return jsonify({"error": "Could not load data"}), 500
        
        # Clean and prepare the data
        df_new.rename(columns={
            'date': 'year',
            'number_cases_sold': 'cases_sold',
            'number_of_girls': 'num_girls'
        }, inplace=True)
        df_new['year'] = df_new['year'].astype(int)
        df_new['troop_id'] = df_new['troop_id'].astype(str).str.strip()
        # Ensure troop_id is formatted as 5-character string with leading zeros for numerical values
        df_new['troop_id'] = df_new['troop_id'].apply(lambda x: f"{int(x):05d}" if x.strip().isdigit() else f"{x:>5}")
        df_new['cookie_type'] = df_new['cookie_type'].str.strip().str.lower()
        # Normalize cookie_type for all downstream logic
        df_new['normalized_cookie_type'] = df_new['cookie_type'].apply(normalize_cookie_type)
        # Debug: Show number of rows for each cookie type in historical data
        cookie_counts = df_new['normalized_cookie_type'].value_counts().to_dict()
        print(f"[DEBUG] Historical data row counts by cookie: {cookie_counts}")

        # Determine the prediction year: use the latest year from the ENTIRE database
        # Get the latest year from the entire database, not just this troop
        global_latest_year = int(df_new['year'].max())
        pred_year = global_latest_year + 1  # Predict for the next year after the latest year in database
        last_year = global_latest_year       # Use the global latest year as "last year"
        
        print(f"[DEBUG] Global latest year in database: {global_latest_year}")
        print(f"[DEBUG] Predicting for year: {pred_year}")
        print(f"[DEBUG] Using year {last_year} as last year data")
        print(f"[DEBUG] ML will use data from years < {pred_year}, SIO will use data from year {last_year}")

        # Check if this troop has data for the latest year
        troop_data = df_new[df_new['troop_id'] == troop_id]
        if troop_data.empty:
            print(f"[DEBUG] No data for troop_id {troop_id}")
            return jsonify({"error": "No data for the specified troop"}), 404
        
        troop_years = sorted(troop_data['year'].unique())
        print(f"[DEBUG] Troop {troop_id} has data for years: {troop_years}")
        
        # For test data, use the latest year (even if empty, so ML can run)
        test = df_new[(df_new['year'] == last_year) & (df_new['troop_id'] == troop_id)]
        # If test is empty, ML predictions will still run using all available data
        # But last year sales based predictions will be null for all cookies
        # (handled in the prediction loop below)

        # Set parameters for prediction.
        lambda_grid = [0.1, 1, 5, 10, 50, 100]
        lambda_default = 10
        k_smooth = 5

        # --- CLUSTERING & PREDICTION LOGIC (unchanged, but use normalized_cookie_type everywhere) ---
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import KFold
        from kneed import KneeLocator
        from tqdm import tqdm
        import numpy as np

        clusters_by_year = {}
        all_predictions = []
        preds_for_rmse = []

        # Use normalized_cookie_type for grouping
        train = df_new[(df_new['year'] >= 2020) & 
                       (df_new['year'] <= last_year) &
                       (df_new['troop_id'] == troop_id)]
        grouped = list(train.groupby(['year', 'SU_Num', 'normalized_cookie_type']))
        for (yr, su, cookie), group in tqdm(grouped, desc=f"Clustering for {pred_year}", leave=False):
            valid = group[(group['cases_sold'] > 0) & (group['num_girls'] > 0)].copy()
            if valid.empty or len(valid) < 3:
                continue
            valid['pga'] = valid['cases_sold'] / valid['num_girls']
            X = valid[['pga']].values
            max_k = min(10, len(X))
            wcss = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
                wcss.append(kmeans.inertia_)
            try:
                knee = KneeLocator(range(1, max_k + 1), wcss, curve='convex', direction='decreasing')
                optimal_k = knee.knee if knee.knee is not None else 1
            except Exception:
                optimal_k = 1
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto").fit(X)
            valid['cluster'] = kmeans.predict(X)
            key = (pred_year, troop_id, cookie)
            if key not in clusters_by_year:
                clusters_by_year[key] = []
            clusters_by_year[key].append(valid[['cases_sold', 'num_girls']])

        # Generate predictions for all active cookies, even if test data is empty
        # If test is empty, we'll use a simplified prediction approach
        print(f"[DEBUG] Test data shape: {test.shape}")
        print(f"[DEBUG] Test empty: {test.empty}")
        if test.empty:
            print(f"[DEBUG] No test data for troop {troop_id} in year {last_year}, using simplified prediction")
            # Get all active cookies that need predictions
            active_cookies = active_df[active_df['status'].str.lower() == 'active']['normalized_cookie_type'].unique()
            print(f"[DEBUG] Active cookies: {active_cookies}")
            
            for cookie in active_cookies:
                # Use historical data for this troop and cookie
                troop_hist = df_new[(df_new['troop_id'] == troop_id) &
                                    (df_new['normalized_cookie_type'] == cookie) &
                                    (df_new['year'] < pred_year)]
                troop_hist = troop_hist[(troop_hist['cases_sold'] > 0) & (troop_hist['num_girls'] > 0)]
                print(f"[DEBUG] Cookie {cookie}: troop_hist shape {troop_hist.shape}")
                
                if not troop_hist.empty:
                    # Calculate average cases per girl
                    avg_cases_per_girl = (troop_hist['cases_sold'] / troop_hist['num_girls']).mean()
                    predicted_cases = avg_cases_per_girl * input_num_girls
                    print(f"[DEBUG] Cookie {cookie}: avg_cases_per_girl={avg_cases_per_girl}, predicted_cases={predicted_cases}")
                    print(f"[DEBUG] Cookie {cookie}: years available in troop_hist: {sorted(troop_hist['year'].unique())}")
                    
                    all_predictions.append({
                        "troop_id": troop_id,
                        "cookie_type": cookie,
                        "actual": None,
                        "predicted": predicted_cases,
                        "method": "avg_pga_no_test_data",
                        "candidate_mse": None,
                        "cluster_std": None,
                        "su": None,
                        "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                        "source": "model"
                    })
                else:
                    # No historical data for this cookie
                    print(f"[DEBUG] Cookie {cookie}: no historical data")
                    all_predictions.append({
                        "troop_id": troop_id,
                        "cookie_type": cookie,
                        "actual": None,
                        "predicted": None,
                        "method": "no_data",
                        "candidate_mse": None,
                        "cluster_std": None,
                        "su": None,
                        "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                        "source": "missing"
                    })
            print(f"[DEBUG] Simplified predictions generated: {len(all_predictions)}")
        else:
            # Use normalized_cookie_type for test grouping
            for (t, cookie), group_test in tqdm(test.groupby(['troop_id', 'normalized_cookie_type']), desc=f"Prediction for {pred_year}", leave=False):
                test_row = group_test.iloc[0]
                su_val = test_row.get("SU_Num", None)
                key_prefix = (pred_year, t, cookie)
                training_dfs = clusters_by_year.get(key_prefix, [])
                cluster_df = pd.concat(training_dfs, ignore_index=True) if training_dfs else pd.DataFrame()
                cluster_std = cluster_df['cases_sold'].std() if not cluster_df.empty else None

                # ... (rest of candidate logic, unchanged, but use normalized_cookie_type everywhere) ...
                # Candidate 1: Ridge with clustering.
                ridge_cluster_pred, mse_cluster = None, float('inf')
                ridge_troop_pred, mse_troop, lambda_cv = None, float('inf'), None
                lin_pred, mse_lin = None, float('inf')
                pga_last_pred, mse_pga_last = None, float('inf')
                pga_avg_pred, mse_pga_avg = None, float('inf')
                su_pred, mse_su = None, float('inf')

                if not cluster_df.empty and len(cluster_df) >= 2:
                    X = np.c_[np.ones(len(cluster_df)), cluster_df['num_girls'].values]
                    y = cluster_df['cases_sold'].values.reshape(-1, 1)
                    kf = KFold(n_splits=min(len(cluster_df), 5), shuffle=True, random_state=42)
                    best_lambda = lambda_default
                    best_mse = float('inf')
                    for lam in lambda_grid:
                        mses = []
                        for train_idx, val_idx in kf.split(X):
                            X_tr, X_val = X[train_idx], X[val_idx]
                            y_tr, y_val = y[train_idx], y[val_idx]
                            I = np.eye(X.shape[1])
                            I[0, 0] = 0
                            beta = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                            y_val_pred = X_val @ beta
                            mses.append(mean_squared_error(y_val, y_val_pred))
                        avg_mse = np.mean(mses)
                        if avg_mse < best_mse:
                            best_mse = avg_mse
                            best_lambda = lam
                    alpha = len(cluster_df) / (len(cluster_df) + k_smooth)
                    lambda_final = alpha * best_lambda + (1 - alpha) * lambda_default
                    I = np.eye(X.shape[1])
                    I[0, 0] = 0
                    beta = np.linalg.inv(X.T @ X + lambda_final * I).dot(X.T @ y)
                    ridge_cluster_pred = np.array([1, input_num_girls]) @ beta
                    mse_cluster = mean_squared_error(y, X @ beta)

                # Candidate 2: Ridge on troop only.
                troop_hist = df_new[(df_new['troop_id'] == t) &
                                    (df_new['normalized_cookie_type'] == cookie) &
                                    (df_new['year'] < pred_year)]
                troop_hist = troop_hist[(troop_hist['cases_sold'] > 0) & (troop_hist['num_girls'] > 0)]
                n_train = len(troop_hist)
                print(f"[DEBUG] Complex ML for {cookie}: troop_hist has {n_train} training points from years {sorted(troop_hist['year'].unique()) if not troop_hist.empty else 'none'}")
                if n_train > 1:
                    X_troop = np.c_[np.ones(n_train), troop_hist['num_girls'].values]
                    y_troop = troop_hist['cases_sold'].values.reshape(-1, 1)
                    if n_train == 2:
                        best_mse = float('inf')
                        for lam in lambda_grid:
                            X_tr, X_val = X_troop[:1], X_troop[1:]
                            y_tr, y_val = y_troop[:1], y_troop[1:]
                            I = np.eye(X_troop.shape[1])
                            I[0, 0] = 0
                            beta_temp = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                            y_pred_temp = X_val @ beta_temp
                            mse_val = mean_squared_error(y_val, y_pred_temp)
                            if mse_val < best_mse:
                                best_mse = mse_val
                                lambda_cv = lam
                    elif n_train >= 3:
                        kf = KFold(n_splits=min(n_train, 3), shuffle=True, random_state=42)
                        best_mse = float('inf')
                        for lam in lambda_grid:
                            mses = []
                            for train_idx, val_idx in kf.split(X_troop):
                                X_tr, X_val = X_troop[train_idx], X_troop[val_idx]
                                y_tr, y_val = y_troop[train_idx], y_troop[val_idx]
                                I = np.eye(X_troop.shape[1])
                                I[0, 0] = 0
                                beta_temp = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                                y_pred_temp = X_val @ beta_temp
                                mses.append(mean_squared_error(y_val, y_pred_temp))
                            avg_mse = np.mean(mses)
                            if avg_mse < best_mse:
                                best_mse = avg_mse
                                lambda_cv = lam
                    else:
                        lambda_cv = lambda_default

                    if lambda_cv is not None:
                        alpha = n_train / (n_train + k_smooth)
                        lambda_final_troop = alpha * lambda_cv + (1 - alpha) * lambda_default
                        I = np.eye(X_troop.shape[1])
                        I[0, 0] = 0
                        beta = np.linalg.inv(X_troop.T @ X_troop + lambda_final_troop * I).dot(X_troop.T @ y_troop)
                        ridge_troop_pred = np.array([1, input_num_girls]) @ beta
                        mse_troop = mean_squared_error(y_troop, X_troop @ beta)
                else:
                    ridge_troop_pred = None
                    mse_troop = float('inf')

                # Candidate 3: Linear Regression.
                if n_train >= 2:
                    model = LinearRegression().fit(troop_hist[['num_girls']], troop_hist['cases_sold'])
                    lin_pred = model.predict([[input_num_girls]])[0]
                    mse_lin = mean_squared_error(troop_hist['cases_sold'],
                                                 model.predict(troop_hist[['num_girls']]))
                # Candidate 4: Last Year PGA Prediction. (COMMENTED OUT - No longer used to avoid duplication with SIO)
                # if not troop_hist.empty:
                #     last_year = troop_hist['year'].max()
                #     last_row = troop_hist[troop_hist['year'] == last_year].iloc[0]
                #     pga_last = last_row['cases_sold'] / last_row['num_girls']
                #     pga_last_pred = pga_last * input_num_girls
                #     mse_pga_last = mean_squared_error([last_row['cases_sold']],
                #                                       [pga_last * last_row['num_girls']])
                # Candidate 5: Average PGA Prediction. (COMMENTED OUT - No longer used to avoid duplication with SIO)
                # if not troop_hist.empty:
                #     avg_pga = (troop_hist['cases_sold'] / troop_hist['num_girls']).mean()
                #     pga_avg_pred = avg_pga * input_num_girls
                #     mse_pga_avg = mean_squared_error(troop_hist['cases_sold'],
                #                                      troop_hist['num_girls'] * avg_pga)
                # Candidate 6: SU-level Ridge without clustering.
                su_data = df_new[(df_new['SU_Num'] == test_row['SU_Num']) &
                                 (df_new['normalized_cookie_type'] == cookie) &
                                 (df_new['year'] < pred_year)]
                su_data = su_data[(su_data['cases_sold'] > 0) & (su_data['num_girls'] > 0)]
                if len(su_data) >= 3:
                    X = np.c_[np.ones(len(su_data)), su_data['num_girls'].values]
                    y = su_data['cases_sold'].values.reshape(-1, 1)
                    kf = KFold(n_splits=min(len(su_data), 5), shuffle=True, random_state=42)
                    best_lambda = lambda_default
                    best_mse = float('inf')
                    for lam in lambda_grid:
                        mses = []
                        for train_idx, val_idx in kf.split(X):
                            X_tr, X_val = X[train_idx], X[val_idx]
                            y_tr, y_val = y[train_idx], y[val_idx]
                            I = np.eye(X.shape[1])
                            I[0, 0] = 0
                            beta = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                            y_val_pred = X_val @ beta
                            mses.append(mean_squared_error(y_val, y_val_pred))
                        avg_mse = np.mean(mses)
                        if avg_mse < best_mse:
                            best_mse = avg_mse
                            best_lambda = lam
                    I = np.eye(X.shape[1])
                    I[0, 0] = 0
                    beta = np.linalg.inv(X.T @ X + best_lambda * I).dot(X.T @ y)
                    su_pred = np.array([1, input_num_girls]) @ beta
                    mse_su = mean_squared_error(y, X @ beta)
                # Choose the best candidate prediction.
                # Exclude PGA-based methods to avoid duplication with SIO predictions
                candidates = [
                    ('cluster_ridge', ridge_cluster_pred, mse_cluster),
                    ('troop_ridge', ridge_troop_pred, mse_troop),
                    ('linreg', lin_pred, mse_lin),
                    ('su_ridge', su_pred, mse_su)
                ]
                valid_candidates = [(name, pred, err) for name, pred, err in candidates if pred is not None and not np.isnan(pred)]
                print(f"[DEBUG] All ML candidates for {cookie}: {[(name, round(float(pred), 2) if pred is not None else None, round(float(err), 2) if err != float('inf') else 'inf') for name, pred, err in candidates]}")
                if valid_candidates:
                    best_method, best_pred, best_mse = min(valid_candidates, key=lambda x: x[2])
                    print(f"[DEBUG] Valid ML candidates for {cookie}: {[(name, round(float(pred), 2), round(float(err), 2)) for name, pred, err in valid_candidates]}")
                    print(f"[DEBUG] ML winner: {best_method} with prediction {float(best_pred):.2f} (MSE: {float(best_mse):.2f})")
                    preds_for_rmse.append(best_pred)
                    
                    # Add the successful ML prediction to all_predictions
                    all_predictions.append({
                        "troop_id": troop_id,
                        "cookie_type": cookie,
                        "actual": test_row['cases_sold'],
                        "predicted": float(best_pred),
                        "method": best_method,
                        "candidate_mse": float(best_mse),
                        "cluster_std": cluster_std,
                        "su": test_row.get("SU_Num", None),
                        "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                        "source": "model"
                    })
                else:
                    print(f"[DEBUG] No valid ML candidates for {cookie} - all returned None or NaN")
                    all_predictions.append({
                        "troop_id": troop_id,
                        "cookie_type": cookie,
                        "actual": test_row['cases_sold'],
                        "predicted": None,
                        "method": "no_valid_candidates",
                        "candidate_mse": None,
                        "cluster_std": cluster_std,
                        "su": test_row.get("SU_Num", None),
                        "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                        "source": "missing"
                    })
        # Fallback: if no candidate predictions were generated, use the test data's PGA.
        if not all_predictions:
            if not test.empty:
                for (t, cookie), group_test in test.groupby(['troop_id', 'normalized_cookie_type']):
                    test_row = group_test.iloc[0]
                    pga = test_row['cases_sold'] / test_row['num_girls']
                    fallback_pred = pga * input_num_girls
                    all_predictions.append({
                        "troop_id": troop_id,
                        "cookie_type": cookie,
                        "actual": test_row['cases_sold'],
                        "predicted": fallback_pred,
                        "method": "fallback_pga",
                        "candidate_mse": None,
                        "cluster_std": None,
                        "su": test_row.get("SU_Num", None),
                        "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                        "source": "fallback"
                    })
            else:
                # No test data and no predictions generated - this should not happen with the simplified logic above
                print(f"[DEBUG] No predictions generated and no test data for troop {troop_id}")
        # Create final predictions without intervals and add last year based predictions
        final_predictions = []
        for pred in all_predictions:
            cookie = pred["cookie_type"]
            predicted_val = pred["predicted"]
            
            # Skip if no prediction was made
            if predicted_val is None:
                final_predictions.append({
                    "cookie_type": cookie,
                    "predicted_cases": None,
                    "last_year_sales": None,
                    "last_year_based_prediction": None,
                    "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                    "source": "missing"
                })
                continue
            
            predicted_val = float(predicted_val)
            
            # Get last year sales for this cookie type
            last_year_sales = None
            last_year_girls = None
            
            # Look for actual last year data for this troop and cookie
            last_year_data = df_new[(df_new['year'] == last_year) & 
                                   (df_new['troop_id'] == troop_id) & 
                                   (df_new['normalized_cookie_type'] == cookie)]
            
            # Only set last year sales if we have actual historical data with valid sales and girls
            if not last_year_data.empty:
                last_year_sales_sum = last_year_data['cases_sold'].sum()
                last_year_girls_sum = last_year_data['num_girls'].sum()  # Sum all girls for this troop and cookie
                
                # Only set values if we have valid data (sales > 0 and girls > 0)
                if last_year_girls_sum > 0 and last_year_sales_sum > 0:
                    last_year_sales = last_year_sales_sum
                    last_year_girls = last_year_girls_sum
                    print(f"[DEBUG] Troop {troop_id} - Last year sales: {last_year_sales}, girls: {last_year_girls} for cookie {cookie}")
                else:
                    # No valid last year data (either no girls or no sales)
                    print(f"[DEBUG] Troop {troop_id} - No valid last year data for cookie {cookie} (sales: {last_year_sales_sum}, girls: {last_year_girls_sum})")
            else:
                # No last year data available
                print(f"[DEBUG] Troop {troop_id} - No last year data for cookie {cookie}")
            
            # Calculate last year based prediction
            last_year_based_prediction = None
            if last_year_sales is not None and last_year_girls is not None and last_year_girls > 0:
                # Scale last year's sales based on the ratio of girls
                girls_ratio = input_num_girls / last_year_girls
                last_year_based_prediction = last_year_sales * girls_ratio
                print(f"[DEBUG] SIO calculation for {cookie}: {last_year_sales} * ({input_num_girls} / {last_year_girls}) = {last_year_based_prediction:.2f}")
            else:
                print(f"[DEBUG] SIO calculation skipped for {cookie}: last_year_sales={last_year_sales}, last_year_girls={last_year_girls}")
            
            # Always set image_url using normalized cookie type
            image_url = url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True)
            final_predictions.append({
                "cookie_type": cookie,
                "predicted_cases": round(predicted_val, 2),
                "last_year_sales": round(last_year_sales, 2) if last_year_sales is not None else None,
                "last_year_based_prediction": round(last_year_based_prediction, 2) if last_year_based_prediction is not None else None,
                "image_url": image_url,
                "source": "model"
            })
            print(f"[DEBUG] Final result for {cookie}: ML={round(predicted_val, 2)}, SIO={round(last_year_based_prediction, 2) if last_year_based_prediction is not None else None}")
        print(f"[DEBUG] Final predictions before active filter: {final_predictions}")
        
        # --- Active Cookies Logic ---
        # Use already loaded active_df and cookie_image_map
        active_cookies = set(active_df[active_df['status'].str.lower() == 'active']['normalized_cookie_type'])
        print("[DEBUG] active_cookies:", active_cookies)
        
        # If we already have predictions from simplified logic, keep them
        if final_predictions and any(pred.get("predicted_cases") is not None for pred in final_predictions):
            print("[DEBUG] Keeping existing predictions from simplified logic")
            filtered_predictions = final_predictions
        else:
            # Filter and update final_predictions to only include active cookies
            filtered_predictions = []
            for pred in final_predictions:
                cookie = pred["cookie_type"]
                if cookie in active_cookies:
                    filtered_predictions.append(pred)
            print(f"[DEBUG] Filtered predictions: {filtered_predictions}")

            # --- Ensure ALL active cookies are represented ---
            for cookie in active_cookies:
                if not any(p["cookie_type"] == cookie for p in filtered_predictions):
                    filtered_predictions.append({
                        "cookie_type": cookie,
                        "predicted_cases": None,
                        "last_year_sales": None,
                        "last_year_based_prediction": None,
                        "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                        "source": "missing"
                    })

        print(f"[DEBUG] Filtered predictions (completed set): {filtered_predictions}")
        return jsonify(filtered_predictions)
    except Exception as e:
        print("Error in /api/predict:", e)
        return jsonify({"error": str(e)}), 500






# In app.py, modify /api/history endpoint:
@app.route('/api/history/<string:troop_id>')
def get_history(troop_id: str):
    print(f"[DEBUG] /api/history called with troop_id={troop_id}")
    print(f"[DEBUG] troop_id dtype in df: {df['troop_id'].dtype}")
    print(f"[DEBUG] Sample troop_id values: {df['troop_id'].head(5).tolist()}")
    troop_df = df[df['troop_id'].astype(str) == troop_id]
    print(f"[DEBUG] Rows found for troop_id {troop_id}: {len(troop_df)}")
    if troop_df.empty:
        print(f"[DEBUG] No data found for troop_id {troop_id}")
        return jsonify({"error": f"No data for troop_id {troop_id}"}), 404

    sales = troop_df.groupby('period')['number_cases_sold'].sum().reset_index()
    girls = troop_df.groupby('period')['number_of_girls'].mean().reset_index()

    su = None
    suName = None
    if 'SU_Num' in troop_df.columns and 'SU_Name' in troop_df.columns:
        su_val = troop_df['SU_Num'].iloc[0]
        # Handle both string and float SU_Num values
        if isinstance(su_val, str):
            su = int(float(su_val))
        elif isinstance(su_val, float):
            su = int(su_val)
        else:
            su = int(su_val)
        suName = troop_df['SU_Name'].iloc[0]

    return jsonify({
        "totalSalesByPeriod": [{"period": int(r['period']), "totalSales": r['number_cases_sold']} for _, r in sales.iterrows()],
        "girlsByPeriod": [{"period": int(r['period']), "numberOfGirls": r['number_of_girls']} for _, r in girls.iterrows()],
        "su": su,
        "suName": suName
    })


@app.route('/api/cookie_breakdown/<string:troop_id>')
def get_breakdown(troop_id: str):
    print(f"[DEBUG] /api/cookie_breakdown called with troop_id={troop_id}")
    print(f"[DEBUG] troop_id dtype in df: {df['troop_id'].dtype}")
    print(f"[DEBUG] Sample troop_id values: {df['troop_id'].head(5).tolist()}")
    troop_df = df[df['troop_id'].astype(str) == troop_id]
    print(f"[DEBUG] Rows found for troop_id {troop_id}: {len(troop_df)}")
    if troop_df.empty:
        print(f"[DEBUG] No data found for troop_id {troop_id}")
        return jsonify([])

    grouped = troop_df.groupby(['period', 'canonical_cookie_type'])['number_cases_sold'].sum().reset_index()
    pivoted = grouped.pivot(index='period', columns='canonical_cookie_type', values='number_cases_sold').fillna(0)
    pivoted.reset_index(inplace=True)

    return jsonify(pivoted.to_dict(orient='records'))


@app.route('/api/su_search')
def su_search():
    query = request.args.get('q', '').strip()
    
    # Filter out SU numbers that contain letters and ensure they are numeric
    df_clean = df.copy()
    df_clean['SU_Num'] = df_clean['SU_Num'].astype(str).str.strip()
    
    # Debug: Check what SU_Num values exist in the database
    print(f"[DEBUG] SU_Num values in database: {df_clean['SU_Num'].unique()[:10]}")
    print(f"[DEBUG] Sample SU_Num values: {df_clean['SU_Num'].head(5).tolist()}")
    
    # Only keep rows where SU_Num contains only digits (including decimal points for float values)
    df_clean = df_clean[df_clean['SU_Num'].str.match(r'^\d+\.?\d*$')]
    
    # Debug: Check what SU_Num values remain after filtering
    print(f"[DEBUG] SU_Num values after filtering: {df_clean['SU_Num'].unique()[:10]}")
    print(f"[DEBUG] Number of rows after filtering: {len(df_clean)}")
    
    # Convert SU_Num to integers (remove decimal points)
    df_clean['SU_Num'] = df_clean['SU_Num'].astype(float).astype(int)
    print(f"[DEBUG] SU_Num values after conversion: {df_clean['SU_Num'].unique()[:10]}")
    
    # If query is empty, return all unique SU numbers
    if not query:
        all_sus = df_clean[['SU_Num', 'SU_Name']].drop_duplicates(subset=['SU_Num']).sort_values('SU_Num')
        results = all_sus.to_dict(orient='records')
        return jsonify(results)
    
    # If query is not a valid number (digits only), return empty array
    if not query.isdigit():
        return jsonify([])

    # Convert query to integer for comparison
    query_int = int(query)
    matches = df_clean[df_clean['SU_Num'] == query_int]
    results = (
        matches[['SU_Num', 'SU_Name']]
        .drop_duplicates(subset=['SU_Num'])
        .sort_values('SU_Num')
        .to_dict(orient='records')
    )
    return jsonify(results)


@app.route('/api/su_history/<string:su_num>')
def su_history(su_num):
    # Load data from database only
    df_new = load_data_from_database()
    df_new['SU_Num'] = df_new['SU_Num'].astype(str).str.strip() if 'SU_Num' in df_new.columns else df_new['SU #'].astype(str).str.strip()
    # Ensure troop_id is formatted as 5-character string with leading zeros for numerical values
    df_new['troop_id'] = df_new['troop_id'].astype(str).str.strip().apply(lambda x: f"{int(x):05d}" if x.strip().isdigit() else f"{x:>5}")
    df_new['canonical_cookie_type'] = df_new['cookie_type'].apply(normalize_cookie_type)
    
    # Convert su_num to integer for consistent comparison
    try:
        su_num_int = int(su_num)
    except ValueError:
        return jsonify({"error": "Invalid SU number format - must be an integer"}), 400
    
    # Handle SU_Num as float in database but integer in API
    # Convert database SU_Num to integers for comparison
    df_new['SU_Num_int'] = pd.to_numeric(df_new['SU_Num'], errors='coerce').astype('Int64')
    df_su = df_new[df_new['SU_Num_int'] == su_num_int]
    
    if df_su.empty:
        return jsonify({"error": "No data"}), 404

    # Average number of girls per year across all troops
    girls_by_year = df_su.groupby('period')['number_of_girls'].mean().reset_index()

    # Step 1: Get total cases sold per troop per year (sum over cookie types)
    troop_sales = df_su.groupby(['period', 'troop_id'])['number_cases_sold'].sum().reset_index()

    # Step 2: Get average total sales per troop per year
    sales_by_year = troop_sales.groupby('period')['number_cases_sold'].mean().reset_index()
    sales_by_year.rename(columns={'number_cases_sold': 'avgSales'}, inplace=True)

    # Scatter plot data (per cookie type) - include year/period field
    scatter = df_su[['number_of_girls', 'number_cases_sold', 'canonical_cookie_type', 'period']].dropna().to_dict(orient='records')

    return jsonify({
        "girlsByYear": [
            {"period": int(r['period']), "avgGirls": r['number_of_girls']} for _, r in girls_by_year.iterrows()
        ],
        "salesByYear": [
            {"period": int(r['period']), "avgSales": r['avgSales']} for _, r in sales_by_year.iterrows()
        ],
        "scatterData": scatter
    })


@app.route('/api/su_scatter_regression/<string:su_num>')
def su_scatter_regression(su_num):
    from scipy.stats import linregress
    # Load data from database only
    df_new = load_data_from_database()
    df_new['SU_Num'] = df_new['SU_Num'].astype(str).str.strip() if 'SU_Num' in df_new.columns else df_new['SU #'].astype(str).str.strip()
    # Ensure troop_id is formatted as 5-character string with leading zeros for numerical values
    df_new['troop_id'] = df_new['troop_id'].astype(str).str.strip().apply(lambda x: f"{int(x):05d}" if x.strip().isdigit() else f"{x:>5}")
    
    # Convert su_num to integer for consistent comparison
    try:
        su_num_int = int(su_num)
    except ValueError:
        return jsonify({"error": "Invalid SU number format - must be an integer"}), 400
    
    # Handle SU_Num as float in database but integer in API
    # Convert database SU_Num to integers for comparison
    df_new['SU_Num_int'] = pd.to_numeric(df_new['SU_Num'], errors='coerce').astype('Int64')
    df_su = df_new[df_new['SU_Num_int'] == su_num_int]
    filtered = df_su.dropna(subset=['number_of_girls', 'number_cases_sold'])
    if filtered.empty or filtered['number_of_girls'].nunique() < 2:
        return jsonify({"line": [], "lower": [], "upper": []})
    # Optional: remove outliers from 'number_cases_sold'
    q1 = filtered['number_cases_sold'].quantile(0.25)
    q3 = filtered['number_cases_sold'].quantile(0.75)
    iqr = q3 - q1
    filtered = filtered[
        (filtered['number_cases_sold'] >= q1 - 1.5 * iqr) &
        (filtered['number_cases_sold'] <= q3 + 1.5 * iqr)
    ]
    if filtered.empty or filtered['number_of_girls'].nunique() < 2:
        return jsonify({"line": [], "lower": [], "upper": []})
    # Run linear regression
    x = filtered['number_of_girls']
    y = filtered['number_cases_sold']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # Build line arrays
    x_vals = sorted(set(x))
    line = []
    lower = []
    upper = []
    for xi in x_vals:
        pred = slope * xi + intercept
        margin = 2 * std_err
        line.append({"x": xi, "y": pred})
        lower.append({"x": xi, "y": pred - margin})
        upper.append({"x": xi, "y": pred + margin})
    return jsonify({"line": line, "lower": lower, "upper": upper})
@app.route('/api/regression/<string:troop_id>')
def regression(troop_id: str):
    # Filter data for the given troop ID
    troop_df = df[df['troop_id'].astype(str) == troop_id]
    if troop_df.empty:
        return jsonify({"error": "No data found for troop"}), 404

    # Extract x and y values
    x = troop_df['number_of_girls']
    y = troop_df['number_cases_sold']

    # Perform a simple linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Create regression line points over the range of x
    x_min, x_max = x.min(), x.max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = slope * x_vals + intercept

    # Compute a simple confidence band: ±2 * std_err
    margin = 2 * std_err
    lower_band = y_vals - margin
    upper_band = y_vals + margin

    # Combine band data into a single array
    band_data = []
    for xv, lb, ub in zip(x_vals, lower_band, upper_band):
        band_data.append({
            "number_of_girls": float(xv),
            "lower": float(lb),
            "upper": float(ub)
        })

    # Prepare scatter data points from the raw data
    scatter_data = [
        {"number_of_girls": float(ng), "number_cases_sold": float(cs)}
        for ng, cs in zip(x, y)
    ]

    # Prepare regression line data points
    line_data = [
        {"number_of_girls": float(x_val), "number_cases_sold": float(y_val)}
        for x_val, y_val in zip(x_vals, y_vals)
    ]

    return jsonify({
        "scatter": scatter_data,
        "regression_line": line_data,
        "band": band_data
    })
@app.route('/api/regression/<string:su_num>')
def regression_su(su_num):
    # Convert su_num to integer for consistent comparison
    try:
        su_num_int = int(su_num)
    except ValueError:
        return jsonify({"error": "Invalid SU number format - must be an integer"}), 400
    
    # Handle SU_Num as float in database but integer in API
    # Convert database SU_Num to integers for comparison
    df['SU_Num_int'] = pd.to_numeric(df['SU_Num'], errors='coerce').astype('Int64')
    su_df = df[df['SU_Num_int'] == su_num_int]
    if su_df.empty:
        return jsonify({"error": "No data found for SU"}), 404

    # Extract x and y values for regression
    x = su_df['number_of_girls']
    y = su_df['number_cases_sold']

    # Ensure there is enough variation in x for regression
    if x.nunique() < 2:
        return jsonify({"error": "Not enough data to perform regression"}), 400

    # Run linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Generate regression line data over the range of x values
    x_min, x_max = x.min(), x.max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = slope * x_vals + intercept

    # Compute a confidence band: ± 2×std_err
    margin = 2 * std_err
    lower_band = y_vals - margin
    upper_band = y_vals + margin

    # Build the confidence band data points
    band_data = []
    for xv, lb, ub in zip(x_vals, lower_band, upper_band):
        band_data.append({
            "number_of_girls": float(xv),
            "lower": float(lb),
            "upper": float(ub)
        })

    # Prepare the scatter (raw) data points
    scatter_data = [
        {"number_of_girls": float(ng), "number_cases_sold": float(cs)}
        for ng, cs in zip(x, y)
    ]

    # Prepare the regression line data points
    regression_line = [
        {"number_of_girls": float(x_val), "number_cases_sold": float(y_val)}
        for x_val, y_val in zip(x_vals, y_vals)
    ]

    return jsonify({
        "scatter": scatter_data,
        "regression_line": regression_line,
        "band": band_data
    })

@app.route('/api/su_predict', methods=['POST'])
def su_predict():
    try:
        data = request.get_json()
        print("[DEBUG] /api/su_predict received:", data)
        su_num = str(data.get('su_number')).strip()
        num_girls = float(data.get('num_girls'))
        if not su_num or num_girls <= 0:
            return jsonify({"error": "Invalid su_number or num_girls"}), 400

        # Load data from database only
        df_new = load_data_from_database()
        
        # Clean and prepare the data (same as api_predict for consistency)
        df_new.rename(columns={
            'date': 'year',
            'number_cases_sold': 'cases_sold',
            'number_of_girls': 'num_girls'
        }, inplace=True)
        df_new['year'] = df_new['year'].astype(int)
        df_new['period'] = df_new['period'].astype(int)
        df_new['troop_id'] = df_new['troop_id'].astype(str).str.strip()
        # Ensure troop_id is formatted as 5-character string with leading zeros for numerical values
        df_new['troop_id'] = df_new['troop_id'].apply(lambda x: f"{int(x):05d}" if x.strip().isdigit() else f"{x:>5}")
        df_new['SU_Num'] = df_new['SU_Num'].astype(str).str.strip() if 'SU_Num' in df_new.columns else df_new['SU #'].astype(str).str.strip()
        df_new['cookie_type'] = df_new['cookie_type'].str.strip().str.lower()
        df_new['normalized_cookie_type'] = df_new['cookie_type'].apply(normalize_cookie_type)

        # Filter for the selected SU
        su_col = 'SU_Num' if 'SU_Num' in df_new.columns else 'SU #' if 'SU #' in df_new.columns else None
        if su_col is None:
            return jsonify({"error": "SU column not found in data"}), 500
        
        # Convert su_num to integer for consistent comparison
        try:
            su_num_int = int(su_num)
        except ValueError:
            return jsonify({"error": "Invalid SU number format - must be an integer"}), 400
        
        # Debug: Show available SU_Num values
        print(f"[DEBUG] Available SU_Num values: {df_new['SU_Num'].unique()[:10]}")
        print(f"[DEBUG] Looking for SU_Num: {su_num_int}")
        
        # Handle SU_Num as float in database but integer in API
        # Convert database SU_Num to integers for comparison
        df_new['SU_Num_int'] = pd.to_numeric(df_new['SU_Num'], errors='coerce').astype('Int64')
        su_data = df_new[df_new['SU_Num_int'] == su_num_int]
        
        if su_data.empty:
            return jsonify({"error": "No data for the specified SU"}), 404

        # Add normalized_cookie_type column to df_new for prediction logic
        df_new['normalized_cookie_type'] = df_new['cookie_type'].apply(normalize_cookie_type)
        su_data = df_new[df_new['SU_Num_int'] == su_num_int]  # Refresh su_data with normalized column
        
        # Debug: Check if the data is loaded correctly
        print(f"[DEBUG] df_new shape: {df_new.shape}")
        print(f"[DEBUG] df_new SU_Num_int unique values: {df_new['SU_Num_int'].unique()[:10]}")
        print(f"[DEBUG] df_new 2025 data for SU 153: {len(df_new[(df_new['year'] == 2025) & (df_new['SU_Num_int'] == 153)])} rows")
        print(f"[DEBUG] df_new 2025 cookie types for SU 153: {df_new[(df_new['year'] == 2025) & (df_new['SU_Num_int'] == 153)]['cookie_type'].unique()}")

        # Load active_cookies and build image map
        engine = get_database_connection()
        active_df = pd.read_sql("SELECT * FROM active_cookies", engine)
        active_df['normalized_cookie_type'] = active_df['cookie_type'].apply(normalize_cookie_type)
        cookie_image_map = dict(zip(active_df['normalized_cookie_type'], active_df['image_filename']))

        # Create reverse mapping from normalized to original cookie types
        normalized_to_original = {}
        for _, row in active_df.iterrows():
            normalized = normalize_cookie_type(row['cookie_type'])
            normalized_to_original[normalized] = row['cookie_type']
        
        print(f"[DEBUG] Normalized to original mapping: {normalized_to_original}")
        print(f"[DEBUG] SU data shape: {su_data.shape}")
        print(f"[DEBUG] SU data cookie types: {su_data['normalized_cookie_type'].unique()}")

        # Dynamic year detection - use the latest year from the ENTIRE database
        # Get the latest year from the entire database, not just this SU
        global_latest_year = int(df_new['year'].max())
        pred_year = global_latest_year + 1  # Predict for the next year after the latest year in database
        last_year = global_latest_year       # Use the global latest year as "last year"
        
        print(f"[DEBUG] Global latest year in database: {global_latest_year}")
        print(f"[DEBUG] SU {su_num} - Predicting for year: {pred_year}")
        print(f"[DEBUG] SU {su_num} - Using year {last_year} as last year data")

        # Build clustering context for SU-level complex ML (mirrors troop path)
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error

        train_su = df_new[(df_new['year'] >= 2020) &
                          (df_new['year'] <= last_year) &
                          (df_new['SU_Num_int'] == su_num_int)]
        clusters_by_year_su: Dict[Any, List[pd.DataFrame]] = {}
        grouped_su = list(train_su.groupby(['year', 'SU_Num_int', 'normalized_cookie_type']))
        for (yr, su_int, cookie), group in grouped_su:
            valid = group[(group['cases_sold'] > 0) & (group['num_girls'] > 0)].copy()
            if valid.empty or len(valid) < 3:
                continue
            valid['pga'] = valid['cases_sold'] / valid['num_girls']
            X = valid[['pga']].values
            max_k = min(10, len(X))
            wcss = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
                wcss.append(kmeans.inertia_)
            try:
                from kneed import KneeLocator
                knee = KneeLocator(range(1, max_k + 1), wcss, curve='convex', direction='decreasing')
                optimal_k = knee.knee if knee.knee is not None else 1
            except Exception:
                optimal_k = 1
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto").fit(X)
            valid['cluster'] = kmeans.predict(X)
            key = (pred_year, su_int, cookie)
            clusters_by_year_su.setdefault(key, []).append(valid[['cases_sold', 'num_girls']])

        # Define test set for SU = last_year rows for this SU
        test_su = df_new[(df_new['year'] == last_year) & (df_new['SU_Num_int'] == su_num_int)]
        print(f"[DEBUG] SU {su_num} - Test data shape: {test_su.shape} (year={last_year})")

        # Active cookies set (for coverage in fallback)
        active_cookies_set = set(active_df[active_df['status'].str.lower() == 'active']['normalized_cookie_type'])
        print(f"[DEBUG] SU {su_num} - Active cookies: {sorted(active_cookies_set)}")

        all_predictions: List[Dict[str, Any]] = []

        if test_su.empty:
            print(f"[DEBUG] SU {su_num} - No last-year rows; using simplified average-per-girl fallback")
            for cookie in active_cookies_set:
                su_hist = su_data[(su_data['normalized_cookie_type'] == cookie) & (su_data['year'] < pred_year)]
                su_hist = su_hist[(su_hist['cases_sold'] > 0) & (su_hist['num_girls'] > 0)]
                if not su_hist.empty and su_hist['num_girls'].sum() > 0:
                    avg_pga = (su_hist['cases_sold'].sum() / su_hist['num_girls'].sum())
                    best_pred = max(0.0, avg_pga * num_girls)
                    best_method = 'average_fallback'
                else:
                    best_pred = None
                    best_method = 'missing'

                # SIO from last-year totals (will be None because test_su is empty)
                last_year_data = df_new[(df_new['year'] == last_year) &
                                        (df_new['SU_Num_int'] == su_num_int) &
                                        (df_new['normalized_cookie_type'] == cookie)]
                last_year_sales = float(last_year_data['cases_sold'].sum()) if not last_year_data.empty else None
                last_year_girls = float(last_year_data['num_girls'].sum()) if not last_year_data.empty else None
                if last_year_sales and last_year_girls and last_year_girls > 0:
                    last_year_based_prediction = last_year_sales * (num_girls / last_year_girls)
                else:
                    last_year_based_prediction = None

                image_url = url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True)
                all_predictions.append({
                    'cookie_type': cookie,
                    'predicted_cases': round(float(best_pred), 2) if best_pred is not None else None,
                    'last_year_sales': round(float(last_year_sales), 2) if last_year_sales is not None else None,
                    'last_year_based_prediction': round(float(last_year_based_prediction), 2) if last_year_based_prediction is not None else None,
                    'image_url': image_url,
                    'source': best_method
                })
                print(f"[DEBUG] SU {su_num} - Fallback prediction for {cookie}: pred={best_pred}, SIO={last_year_based_prediction}")
        else:
            # Iterate over cookies present in test year for this SU
            test_cookies = sorted(test_su['normalized_cookie_type'].unique())
            print(f"[DEBUG] SU {su_num} - Test cookies: {test_cookies}")
            for cookie in test_cookies:
                group_test = test_su[test_su['normalized_cookie_type'] == cookie]
                test_row = group_test.iloc[0]

                # Candidates
                ridge_cluster_pred, mse_cluster = None, float('inf')
                su_ridge_pred, mse_su, lambda_cv_su = None, float('inf'), None
                lin_pred, mse_lin = None, float('inf')

                # Candidate 1: cluster_ridge using SU clusters
                key = (pred_year, su_num_int, cookie)
                training_dfs = clusters_by_year_su.get(key, [])
                cluster_df = pd.concat(training_dfs, ignore_index=True) if training_dfs else pd.DataFrame()
                cluster_std = cluster_df['cases_sold'].std() if not cluster_df.empty else None
                if not cluster_df.empty and len(cluster_df) >= 2:
                    X = np.c_[np.ones(len(cluster_df)), cluster_df['num_girls'].values]
                    y = cluster_df['cases_sold'].values.reshape(-1, 1)
                    kf = KFold(n_splits=min(len(cluster_df), 5), shuffle=True, random_state=42)
                    best_lambda = 10
                    best_mse_local = float('inf')
                    for lam in [0.1, 1, 5, 10, 50, 100]:
                        mses = []
                        for train_idx, val_idx in kf.split(X):
                            X_tr, X_val = X[train_idx], X[val_idx]
                            y_tr, y_val = y[train_idx], y[val_idx]
                            I = np.eye(X.shape[1]); I[0, 0] = 0
                            beta = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                            y_val_pred = X_val @ beta
                            mses.append(mean_squared_error(y_val, y_val_pred))
                        avg_mse = float(np.mean(mses))
                        if avg_mse < best_mse_local:
                            best_mse_local = avg_mse
                            best_lambda = lam
                    I = np.eye(X.shape[1]); I[0, 0] = 0
                    beta = np.linalg.inv(X.T @ X + best_lambda * I).dot(X.T @ y)
                    ridge_cluster_pred = (np.array([1, num_girls]) @ beta).item()
                    mse_cluster = float(mean_squared_error(y, X @ beta))

                # Candidate 2: su_ridge on SU history
                su_hist = su_data[(su_data['normalized_cookie_type'] == cookie) & (su_data['year'] < pred_year)]
                su_hist = su_hist[(su_hist['cases_sold'] > 0) & (su_hist['num_girls'] > 0)]
                n_train = len(su_hist)
                print(f"[DEBUG] SU {su_num} - {cookie}: su_hist n={n_train}, years={sorted(su_hist['year'].unique()) if n_train>0 else 'none'}")
                if n_train > 1:
                    X_su = np.c_[np.ones(n_train), su_hist['num_girls'].values]
                    y_su = su_hist['cases_sold'].values.reshape(-1, 1)
                    if n_train == 2:
                        best_mse_local = float('inf')
                        best_lambda = 10
                        for lam in [0.1, 1, 5, 10, 50, 100]:
                            X_tr, X_val = X_su[:1], X_su[1:]
                            y_tr, y_val = y_su[:1], y_su[1:]
                            I = np.eye(X_su.shape[1]); I[0, 0] = 0
                            beta_temp = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                            y_pred_temp = X_val @ beta_temp
                            mse_val = float(mean_squared_error(y_val, y_pred_temp))
                            if mse_val < best_mse_local:
                                best_mse_local = mse_val
                                lambda_cv_su = lam
                    else:
                        kf = KFold(n_splits=min(n_train, 3), shuffle=True, random_state=42)
                        best_mse_local = float('inf')
                        best_lambda = 10
                        for lam in [0.1, 1, 5, 10, 50, 100]:
                            mses = []
                            for train_idx, val_idx in kf.split(X_su):
                                X_tr, X_val = X_su[train_idx], X_su[val_idx]
                                y_tr, y_val = y_su[train_idx], y_su[val_idx]
                                I = np.eye(X_su.shape[1]); I[0, 0] = 0
                                beta_temp = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                                y_pred_temp = X_val @ beta_temp
                                mses.append(mean_squared_error(y_val, y_pred_temp))
                            avg_mse = float(np.mean(mses))
                            if avg_mse < best_mse_local:
                                best_mse_local = avg_mse
                                lambda_cv_su = lam
                    if lambda_cv_su is None:
                        lambda_cv_su = 10
                    alpha = n_train / (n_train + 5)
                    lambda_final = alpha * lambda_cv_su + (1 - alpha) * 10
                    I = np.eye(X_su.shape[1]); I[0, 0] = 0
                    beta = np.linalg.inv(X_su.T @ X_su + lambda_final * I).dot(X_su.T @ y_su)
                    su_ridge_pred = (np.array([1, num_girls]) @ beta).item()
                    mse_su = float(mean_squared_error(y_su, X_su @ beta))

                # Candidate 3: Linear Regression on SU rows
                if n_train >= 2:
                    lr = LinearRegression().fit(su_hist[['num_girls']], su_hist['cases_sold'])
                    lin_pred = float(lr.predict([[num_girls]])[0])
                    mse_lin = float(mean_squared_error(su_hist['cases_sold'], lr.predict(su_hist[['num_girls']])))

                candidates = [
                    ('cluster_ridge', ridge_cluster_pred, mse_cluster),
                    ('su_ridge', su_ridge_pred, mse_su),
                    ('linreg', lin_pred, mse_lin)
                ]
                valid_candidates = [(n, p, e) for n, p, e in candidates if p is not None and not np.isnan(p)]
                print(f"[DEBUG] SU {su_num} - {cookie} candidates: {[(n, None if p is None else round(float(p),2), 'inf' if e==float('inf') else round(float(e),2)) for n,p,e in candidates]}")
                if valid_candidates:
                    best_method, best_pred, best_mse = min(valid_candidates, key=lambda x: x[2])
                    print(f"[DEBUG] SU {su_num} - {cookie} winner: {best_method} pred={float(best_pred):.2f} mse={float(best_mse):.2f}")
                else:
                    best_method, best_pred, best_mse = ('no_valid_candidates', None, None)
                    print(f"[DEBUG] SU {su_num} - {cookie} has no valid ML candidates")

                # Compute SIO for SU last-year scaling
                last_year_data = df_new[(df_new['year'] == last_year) &
                                        (df_new['SU_Num_int'] == su_num_int) &
                                        (df_new['normalized_cookie_type'] == cookie)]
                last_year_sales = float(last_year_data['cases_sold'].sum()) if not last_year_data.empty else None
                last_year_girls = float(last_year_data['num_girls'].sum()) if not last_year_data.empty else None
                if last_year_sales and last_year_girls and last_year_girls > 0:
                    last_year_based_prediction = last_year_sales * (num_girls / last_year_girls)
                    print(f"[DEBUG] SU {su_num} - SIO {cookie}: {last_year_sales} * ({num_girls}/{last_year_girls}) = {last_year_based_prediction:.2f}")
                else:
                    last_year_based_prediction = None
                    print(f"[DEBUG] SU {su_num} - SIO {cookie}: no valid last-year totals")

                image_url = url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True)
                all_predictions.append({
                    'cookie_type': cookie,
                    'predicted_cases': round(float(best_pred), 2) if best_pred is not None else None,
                    'last_year_sales': round(float(last_year_sales), 2) if last_year_sales is not None else None,
                    'last_year_based_prediction': round(float(last_year_based_prediction), 2) if last_year_based_prediction is not None else None,
                    'image_url': image_url,
                    'source': best_method
                })
        # --- Cookie Transitions Logic ---
        transitions_df = pd.read_sql("SELECT * FROM cookie_transitions", engine)
        # Normalize cookie names from transitions to match model naming
        if 'New Cookie' in transitions_df.columns:
            transitions_df['New Cookie'] = transitions_df['New Cookie'].apply(lambda x: normalize_cookie_type(str(x)) if pd.notnull(x) else x)
        if 'Replaces Cookie' in transitions_df.columns:
            transitions_df['Replaces Cookie'] = transitions_df['Replaces Cookie'].apply(lambda x: normalize_cookie_type(str(x)) if pd.notnull(x) else x)
        for i in range(1, 6):
            col = f'ShareFrom_{i}'
            if col in transitions_df.columns:
                transitions_df[col] = transitions_df[col].apply(lambda x: normalize_cookie_type(str(x)) if pd.notnull(x) else x)
        historical_cookies = set(su_data['normalized_cookie_type'].unique())
        forecast = {pred["cookie_type"]: float(pred["predicted_cases"]) for pred in all_predictions}
        print(f"[DEBUG] Initial forecast for transitions: {forecast}")

        # NEW: Ensure replaced cookies have forecasts even if not predicted
        replaced_cookies = set(transitions_df['Replaces Cookie'].dropna().unique())
        for rc in replaced_cookies:
            if rc not in forecast:
                hist = su_data[su_data['normalized_cookie_type'] == rc]
                if not hist.empty:
                    avg_pga = (hist['cases_sold'] / hist['num_girls']).mean()
                    pred_val = avg_pga * num_girls
                else:
                    pred_val = 0
                forecast[rc] = pred_val
                all_predictions.append({
                    "cookie_type": rc,
                    "predicted_cases": round(pred_val, 2),
                    "image_url": url_for('static', filename=cookie_image_map.get(rc, "default.png"), _external=True),
                    "source": "fallback"
                })
                print(f"[DEBUG] Added fallback forecast for replaced cookie {rc}: {pred_val}")

        # Proceed with transition logic
        for idx, row in transitions_df.iterrows():
            new_cookie = row['New Cookie']
            replaces_cookie = row['Replaces Cookie']
            print(f"[DEBUG] Transition: {new_cookie} replaces {replaces_cookie}")
            if new_cookie not in historical_cookies:
                base = forecast.get(replaces_cookie, 0)
                if base <= 0:
                    print(f"[DEBUG] No forecast for {replaces_cookie}; skip new cookie {new_cookie}")
                    continue  # Skip creating prediction for this new cookie
                forecast[new_cookie] = base
                print(f"[DEBUG] Base forecast for {new_cookie}: {base}")
                for i in range(1, 6):
                    share_from = row.get(f'ShareFrom_{i}')
                    share_pct = row.get(f'SharePct_{i}')
                    if pd.notnull(share_from) and pd.notnull(share_pct):
                        add_val = forecast.get(share_from, 0) * (share_pct / 100)
                        forecast[new_cookie] += add_val
                        print(f"[DEBUG] Added {add_val} from {share_from} ({share_pct}%)")
                print(f"[DEBUG] Final forecast for {new_cookie}: {forecast[new_cookie]}")
            else:
                print(f"[DEBUG] {new_cookie} has historical data, skipping transition logic")
        
        # Add any new cookies from forecast that are not already in final_predictions
        existing_cookies = {pred["cookie_type"] for pred in all_predictions}
        for cookie, value in forecast.items():
            if cookie not in existing_cookies:
                # Attempt to compute synthetic SIO for new cookies based on transitions
                last_year_sales_syn = 0.0
                last_year_girls_syn = 0.0
                matching_rows = transitions_df[transitions_df['New Cookie'] == cookie]
                if not matching_rows.empty:
                    for _, trow in matching_rows.iterrows():
                        rep = trow.get('Replaces Cookie')
                        if pd.notnull(rep):
                            rep_rows = df_new[(df_new['year'] == last_year) & (df_new['SU_Num_int'] == su_num_int) & (df_new['normalized_cookie_type'] == rep)]
                            sales = rep_rows['cases_sold'].sum() if not rep_rows.empty else 0.0
                            girls = rep_rows['num_girls'].sum() if not rep_rows.empty else 0.0
                            last_year_sales_syn += float(sales)
                            last_year_girls_syn += float(girls)
                            print(f"[DEBUG] Transition SIO base for {cookie} from {rep}: sales={sales}, girls={girls}")
                        for i in range(1, 6):
                            share_from = trow.get(f'ShareFrom_{i}')
                            share_pct = trow.get(f'SharePct_{i}')
                            if pd.notnull(share_from) and pd.notnull(share_pct):
                                sf_rows = df_new[(df_new['year'] == last_year) & (df_new['SU_Num_int'] == su_num_int) & (df_new['normalized_cookie_type'] == share_from)]
                                sf_sales = sf_rows['cases_sold'].sum() if not sf_rows.empty else 0.0
                                sf_girls = sf_rows['num_girls'].sum() if not sf_rows.empty else 0.0
                                add_sales = float(sf_sales) * (float(share_pct) / 100.0)
                                add_girls = float(sf_girls) * (float(share_pct) / 100.0)
                                last_year_sales_syn += add_sales
                                last_year_girls_syn += add_girls
                                print(f"[DEBUG] Transition SIO share for {cookie} from {share_from} ({share_pct}%): add_sales={add_sales}, add_girls={add_girls}")
                else:
                    print(f"[DEBUG] No transitions row found for {cookie}; SIO will remain None if no last-year data")

                if last_year_sales_syn > 0 and last_year_girls_syn > 0:
                    sio_scaled = last_year_sales_syn * (num_girls / last_year_girls_syn)
                else:
                    sio_scaled = None
                print(f"[DEBUG] Transition SIO totals for {cookie}: sales={last_year_sales_syn}, girls={last_year_girls_syn}, scaled={sio_scaled}")

                all_predictions.append({
                    "cookie_type": cookie,
                    "predicted_cases": round(value, 2),
                    "last_year_sales": round(last_year_sales_syn, 2) if last_year_sales_syn > 0 else None,
                    "last_year_based_prediction": round(float(sio_scaled), 2) if sio_scaled is not None else None,
                    "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                    "source": "fallback"
                })
        print(f"[DEBUG] Final predictions after all logic: {all_predictions}")
        active_cookies = set(active_df[active_df['status'].str.lower() == 'active']['normalized_cookie_type'])
        filtered_predictions = [pred for pred in all_predictions if pred["cookie_type"] in active_cookies]

        # Ensure every active cookie is present
        for cookie in active_cookies:
            if not any(p["cookie_type"] == cookie for p in filtered_predictions):
                filtered_predictions.append({
                    "cookie_type": cookie,
                    "predicted_cases": None,
                    "last_year_sales": None,
                    "last_year_based_prediction": None,
                    "image_url": url_for('static', filename=cookie_image_map.get(cookie, "default.png"), _external=True),
                    "source": "missing"
                })

        return jsonify(filtered_predictions)
    except Exception as e:
        print("❌ ERROR in /api/su_predict:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
