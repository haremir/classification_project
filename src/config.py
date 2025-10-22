from pathlib import Path

# Proje yolları
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Veri dosyaları
RAW_DATA_FILE = RAW_DATA_DIR / "Training Dataset.arff"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model dosyaları
RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest.pkl"
DECISION_TREE_MODEL = MODELS_DIR / "decision_tree.pkl"
GRADIENT_BOOSTING_MODEL = MODELS_DIR / "gradient_boosting.pkl"
LOGISTIC_REGRESSION_MODEL = MODELS_DIR / "logistic_regression.pkl"
BEST_MODEL = MODELS_DIR / "best_model.pkl"

# Veri işleme parametreleri
TEST_SIZE = 0.2
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2

# Feature bilgileri
FEATURE_NAMES = [
    'having_IP_Address',
    'URL_Length',
    'Shortining_Service',
    'having_At_Symbol',
    'double_slash_redirecting',
    'Prefix_Suffix',
    'having_Sub_Domain',
    'SSLfinal_State',
    'Domain_registeration_length',
    'Favicon',
    'port',
    'HTTPS_token',
    'Request_URL',
    'URL_of_Anchor',
    'Links_in_tags',
    'SFH',
    'Submitting_to_email',
    'Abnormal_URL',
    'Redirect',
    'on_mouseover',
    'RightClick',
    'popUpWidnow',
    'Iframe',
    'age_of_domain',
    'DNSRecord',
    'web_traffic',
    'Page_Rank',
    'Google_Index',
    'Links_pointing_to_page',
    'Statistical_report'
]

TARGET_NAME = 'Result'

# Model hiperparametreleri
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

DECISION_TREE_PARAMS = {
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'penalty': 'l2',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Cross-validation parametreleri
CV_FOLDS = 5
CV_SCORING = 'f1_weighted'

# Değerlendirme metrikleri
EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Görselleştirme ayarları
FIGURE_SIZE = (10, 6)
DPI = 300
STYLE = 'seaborn-v0_8-darkgrid'

# Sınıf etiketleri
CLASS_LABELS = {
    -1: 'Phishing',
    1: 'Legitimate'
}