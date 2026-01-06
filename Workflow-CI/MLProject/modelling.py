import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Konfigurasi MLflow ke Folder Lokal (PENTING untuk VS Code)
# Supaya hasilnya muncul di dashboard localhost:5000 yang sama dengan yang tadi
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Eksperimen_Basic_Water")

# 2. Load Data
# Pastikan file csv ada di folder yang sama
try:
    df = pd.read_csv('water_potability_preprocessing.csv')
    print("Data berhasil dibaca!")
except FileNotFoundError:
    print("Error: File csv tidak ditemukan.")
    exit()

# Handle jika masih ada missing value (Safety net)
df = df.fillna(df.mean())

# Pisahkan Fitur dan Target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Aktifkan Autolog
mlflow.sklearn.autolog()

# 4. Training Model
print("Sedang melatih model Basic (Autolog)...")

with mlflow.start_run(run_name="Run_Basic_No_Tuning"):
    # Model Random Forest Standar
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Selesai! Akurasi: {acc:.4f}")
    print("Metrik dan model sudah dicatat otomatis oleh Autolog.")