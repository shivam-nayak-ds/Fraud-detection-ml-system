import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR , 'data')

MODEL_DIR = os.path.join(BASE_DIR , 'models')

DATA_PATH = os.path.join(DATA_DIR , 'train_transaction.csv')

MODEL_PATH = os.path.join(MODEL_DIR , 'fraud_model.pkl')

TEST_SIZE = 0.2

RANDOM_STATE = 42


TOP_N_FEATURES = 50
N_ESTIMATORS   = 200
MAX_DEPTH      = 10
MISSING_THRESHOLD = 0.5