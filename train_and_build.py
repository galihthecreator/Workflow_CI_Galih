import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# KONFIGURASI
DAGSHUB_USER = "galihthecreator"
DAGSHUB_REPO = "Eksperimen_SML_Galih"
DOCKER_USER = os.environ.get("DOCKER_USERNAME","galihthecreator")
IMAGE_NAME = 'titanic-model'

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")

#LOAD DATA
def load_data():
    df = pd.read_csv('train_clean.csv')
    
    X = df.drop('Survived', axis = 1, errors ='ignore')
    y =df['Survived']
    return X,y

def train():
    print('Training model...')
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run() as run:
        X,y = load_data()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X,y)

        run_id = run.info.run_id
        print(f'Model Trained. Run ID: {run_id}')

        return run_id
    
if __name__ == "__main__":
    # 1. Train Model
    run_id = train()
    
    # 2. Build Docker Image (Advance Requirement)
    print(f"Building Docker Image for Run ID: {run_id}")
    
    model_uri = f'runs:/{run_id}/model'
    full_image_name =f'{DOCKER_USER}/{IMAGE_NAME}:latest'

    #build
    build_cmd = f'python -m mlflow models build-docker -m {model_uri} -n {full_image_name} --enable-mlserver'
    exit_code = os.system(build_cmd)

    if exit_code != 0:
        raise Exception('Docker build gagal')
    
    print('Docker image built berhasil')
    print('push ke docker hub')

    push_exit_code = os.system(f'docker push {full_image_name}')

    if push_exit_code != 0:
        raise Exception('Docker push gagal, cek apakah sudah login')
    
    print('done! image sudah dipush')