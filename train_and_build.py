import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# KONFIGURASI
DAGSHUB_USER = "galihthecreator"
DAGSHUB_TOKEN = "f2149a203d7a20fac67b54fde0c780dffd410b7c"
DAGSHUB_REPO = "Eksperimen_SML_Galih"
DOCKER_USER = "galihthecreator"
IMAGE_NAME = 'titanic-model'

# setup dagshub mlflow tracking uri
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")

#LOAD DATA
def load_data():
    url = 'https://github.com/galihthecreator/Eksperimen_SML_Galih/blob/main/preprocessing/titanic_cleaned.csv'
    df = pd.read_csv(url)
    df = df.dropna(subnet=['Age','Embarked'])
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df = pd.get_dummies(df, columns=['Embarked'])
    return df.select_dtypes(include=['number'])

def train():
    print('Training model...')
    mlflow.sklearn.autolog(log_models=True)
    df = load_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    mlflow.set_experiment('CI_Docker_Build')

    with mlflow.start_run() as run:
        model = RandomForestClassifier()
        model.fit(X,y)
    
if __name__ == "__main__":
    # 1. Train Model
    run_id = train()
    
    # 2. Build Docker Image (Advance Requirement)
    print(f"Building Docker Image for Run ID: {run_id}")
    
    # Format URI model
    model_uri = f"runs:/{run_id}/model"
    
    # Nama Image Lengkap
    full_image_name = f"{DOCKER_USER}/{IMAGE_NAME}:latest"
    
    # Perintah Build Docker via MLflow
    # Note: Ini akan memanggil 'docker build' di sistem
    os.system(f"mlflow models build-docker -m {model_uri} -n {full_image_name}")
    
    print("Docker Image Built Successfully.")
    print("Pushing to Docker Hub...")
    os.system(f"docker push {full_image_name}")
    print("Done!")
