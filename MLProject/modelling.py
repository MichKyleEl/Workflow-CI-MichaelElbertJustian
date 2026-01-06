import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
try:
    df = pd.read_csv('water_potability_preprocessing.csv')
    print("✅ Data berhasil dibaca!")
except FileNotFoundError:
    try:
        df = pd.read_csv('water_potability_clean.csv')
        print("✅ Data (backup) berhasil dibaca!")
    except:
        print("❌ Error: File dataset tidak ditemukan.")
        exit()

# Handle Missing Values (Safety net)
df = df.fillna(df.mean())

# Pisahkan Fitur dan Target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Aktifkan Autolog
mlflow.sklearn.autolog()

# 3. Training Model
print("⏳ Sedang melatih model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🎉 Selesai! Akurasi: {acc:.4f}")
print("Metrik dan model sudah dicatat otomatis oleh Autolog ke dalam MLflow Run ini.")
