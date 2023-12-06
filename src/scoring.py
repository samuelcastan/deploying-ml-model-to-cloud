import pandas
import joblib

from sklearn.metrics import f1_score

def model_performance():
    pass

if __name__ == '__main__':
    
    # data = pd.read_csv("")
    
    pipeline = joblib.load("model/inference_pipeline.pkl")
    
    model_performance(pipeline, )
