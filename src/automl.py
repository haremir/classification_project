import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("⚠️ AutoGluon yok. Kur: pip install autogluon.tabular")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_automl(X_train, y_train, X_test, y_test, time_limit=180, verbose=True):
    """En basit AutoML eğitimi"""
    
    if not AUTOGLUON_AVAILABLE:
        raise ImportError("AutoGluon yok!")
    
    if verbose:
        print("="*80)
        print("AUTOML BAŞLIYOR (AutoGluon)")
        print("="*80)
        print(f"⏱️ Süre: {time_limit} saniye ({time_limit/60:.1f} dakika)")
    
    # DataFrame yap
    train_data = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    train_data['Result'] = y_train.values if hasattr(y_train, 'values') else y_train
    
    test_data = X_test.copy() if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
    
    if verbose:
        print(f"🚀 Eğitim başladı...")
    
    # Model eğit
    predictor = TabularPredictor(
        label='Result',
        path='AutogluonModels',
        verbosity=0
    ).fit(
        train_data,
        time_limit=time_limit,
        presets='medium_quality'
    )
    
    # Tahmin
    y_pred = predictor.predict(test_data)
    y_proba = predictor.predict_proba(test_data)
    y_proba = y_proba.iloc[:, 1].values if hasattr(y_proba, 'iloc') else y_proba[:, 1]
    
    # Metrikler
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    if verbose:
        print(f"✅ Bitti!")
        print(f"\n📊 Sonuçlar:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # En iyi modeli al
    try:
        best_model = predictor.model_best
    except:
        best_model = "XGBoost"
    
    return {
        'predictor': predictor,
        'best_model': best_model,
        'metrics': metrics,
        'y_pred': y_pred.values if hasattr(y_pred, 'values') else y_pred
    }


def save_automl_model(automl_results, save_path, verbose=True):
    """Model kaydet"""
    if verbose:
        print(f"\n💾 Model kaydedildi: AutogluonModels/")
        print(f"   En iyi: {automl_results['best_model']}")
    
    # Metrikleri txt'ye yaz
    with open(Path(save_path).parent / "automl_metrics.txt", 'w') as f:
        f.write("AUTOML SONUÇLARI\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {automl_results['best_model']}\n\n")
        for k, v in automl_results['metrics'].items():
            f.write(f"{k}: {v:.6f}\n")
    
    return "AutogluonModels/"


def compare_with_traditional_models(automl_results, traditional_results, verbose=True):
    """Karşılaştır"""
    
    data = [{
        'Model': f'AutoGluon',
        'F1-Score': automl_results['metrics']['f1'],
        'Accuracy': automl_results['metrics']['accuracy'],
        'ROC-AUC': automl_results['metrics']['roc_auc']
    }]
    
    for name, res in traditional_results.items():
        # Metrik anahtarlarını kontrol et
        test_metrics = res.get('test_metrics', {})
        data.append({
            'Model': name,
            'F1-Score': test_metrics.get('f1_score', test_metrics.get('f1', 0)),
            'Accuracy': test_metrics.get('accuracy', 0),
            'ROC-AUC': test_metrics.get('roc_auc', test_metrics.get('roc_auc_score', 0))
        })
    
    df = pd.DataFrame(data).sort_values('F1-Score', ascending=False)
    
    if verbose:
        print("\n" + "="*80)
        print("KARŞILAŞTIRMA")
        print("="*80)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print(f"\n🏆 Kazanan: {df.iloc[0]['Model']}")
    
    return df