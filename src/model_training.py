"""
Model Eğitimi Modülü
===================

Bu modül, phishing tespit projesi için çeşitli makine öğrenmesi modellerini
eğitir, değerlendirir ve en iyi modeli seçer.

Fonksiyonlar:
    - train_single_model: Tek bir modeli eğitir
    - calculate_metrics: Model metriklerini hesaplar
    - print_metrics: Metrikleri yazdırır
    - train_all_models: Tüm modelleri eğitir
    - compare_models: Modelleri karşılaştırır
    - save_best_model: En iyi modeli kaydeder
    - save_all_models: Tüm modelleri kaydeder
    - load_model: Kaydedilmiş modeli yükler
    - get_model_summary: Model özeti döner
    - print_model_summary: Model özetini yazdırır
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import warnings
from datetime import datetime

# Scikit-learn modelleri
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

warnings.filterwarnings('ignore')


def train_single_model(X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str, model, verbose: bool = True) -> Dict[str, Any]:
    """
    Tek bir modeli eğitir ve değerlendirir.
    
    Parameters:
        X_train (pd.DataFrame): Train özellikleri
        y_train (pd.Series): Train hedef değişken
        X_test (pd.DataFrame): Test özellikleri
        y_test (pd.Series): Test hedef değişken
        model_name (str): Model ismi
        model: Eğitilecek model
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        Dict[str, Any]: Model sonuçları ve metrikleri
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"MODEL EĞİTİMİ: {model_name.upper()}")
        print(f"{'='*60}")
    
    # Modeli eğit
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Tahminler yap
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Olasılık tahminleri (varsa)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_train_proba = None
        y_test_proba = None
        has_proba = False
    
    # Metrikleri hesapla
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, "Train")
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "Test")
    
    # Sonuçları paketle
    results = {
        'model_name': model_name,
        'model': model,
        'training_time': training_time,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'has_proba': has_proba,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }
    
    if verbose:
        print_metrics(train_metrics, test_metrics, training_time)
    
    return results


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, 
                     y_proba: np.ndarray = None, prefix: str = "") -> Dict[str, float]:
    """
    Model metriklerini hesaplar.
    
    Parameters:
        y_true (pd.Series): Gerçek değerler
        y_pred (np.ndarray): Tahmin edilen değerler
        y_proba (np.ndarray): Olasılık tahminleri (opsiyonel)
        prefix (str): Metrik isimlerine eklenecek önek
        
    Returns:
        Dict[str, float]: Hesaplanan metrikler
    """
    metrics = {
        f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}_precision': precision_score(y_true, y_pred, average='weighted'),
        f'{prefix}_recall': recall_score(y_true, y_pred, average='weighted'),
        f'{prefix}_f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # ROC-AUC (olasılık tahminleri varsa)
    if y_proba is not None:
        try:
            metrics[f'{prefix}_roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics[f'{prefix}_roc_auc'] = None
    
    return metrics


def print_metrics(train_metrics: Dict[str, float], 
                 test_metrics: Dict[str, float], 
                 training_time: float) -> None:
    """
    Model metriklerini yazdırır.
    
    Parameters:
        train_metrics (Dict[str, float]): Train metrikleri
        test_metrics (Dict[str, float]): Test metrikleri
        training_time (float): Eğitim süresi (saniye)
    """
    print(f"\n📊 PERFORMANS METRİKLERİ:")
    print(f"   • Eğitim Süresi: {training_time:.2f} saniye")
    
    print(f"\n   {'Metrik':<15} {'Train':<10} {'Test':<10} {'Fark':<10}")
    print(f"   {'-'*50}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        train_val = train_metrics[f'Train_{metric}']
        test_val = test_metrics[f'Test_{metric}']
        diff = train_val - test_val
        
        print(f"   {metric.capitalize():<15} {train_val:<10.4f} {test_val:<10.4f} {diff:<10.4f}")
    
    # ROC-AUC (varsa)
    if 'Train_roc_auc' in train_metrics and train_metrics['Train_roc_auc'] is not None:
        train_auc = train_metrics['Train_roc_auc']
        test_auc = test_metrics['Test_roc_auc']
        auc_diff = train_auc - test_auc
        
        print(f"   {'ROC-AUC':<15} {train_auc:<10.4f} {test_auc:<10.4f} {auc_diff:<10.4f}")


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    config, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Tüm modelleri eğitir.
    
    Parameters:
        X_train (pd.DataFrame): Train özellikleri
        y_train (pd.Series): Train hedef değişken
        X_test (pd.DataFrame): Test özellikleri
        y_test (pd.Series): Test hedef değişken
        config: Config modülü
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        Dict[str, Dict[str, Any]]: Tüm model sonuçları
    """
    if verbose:
        print(f"\n{'='*80}")
        print("TÜM MODELLERİN EĞİTİMİ")
        print(f"{'='*80}")
        print(f"📊 Veri Bilgileri:")
        print(f"   • Train: {X_train.shape[0]:,} satır x {X_train.shape[1]} özellik")
        print(f"   • Test:  {X_test.shape[0]:,} satır x {X_test.shape[1]} özellik")
    
    # Model sözlüğü
    models = {
        'Random Forest': RandomForestClassifier(**config.RANDOM_FOREST_PARAMS),
        'Decision Tree': DecisionTreeClassifier(**config.DECISION_TREE_PARAMS),
        'Gradient Boosting': GradientBoostingClassifier(**config.GRADIENT_BOOSTING_PARAMS),
        'Logistic Regression': LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    }
    
    if verbose:
        print(f"   • Toplam Model: {len(models)}")
    
    # Tüm modelleri eğit
    all_results = {}
    
    for model_name, model in models.items():
        try:
            results = train_single_model(
                X_train, y_train, X_test, y_test,
                model_name, model, verbose
            )
            all_results[model_name] = results
            
        except Exception as e:
            print(f"❌ {model_name} eğitimi başarısız: {e}")
            continue
    
    if verbose:
        print(f"\n✅ {len(all_results)} model başarıyla eğitildi!")
    
    return all_results


def compare_models(all_results: Dict[str, Dict[str, Any]], 
                  metric: str = 'Test_f1', verbose: bool = True) -> pd.DataFrame:
    """
    Modelleri karşılaştırır ve sıralar.
    
    Parameters:
        all_results (Dict[str, Dict[str, Any]]): Tüm model sonuçları
        metric (str): Karşılaştırma metriği
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        pd.DataFrame: Karşılaştırma tablosu
    """
    if verbose:
        print(f"\n{'='*80}")
        print("MODEL KARŞILAŞTIRMASI")
        print(f"{'='*80}")
    
    # Karşılaştırma tablosu oluştur
    comparison_data = []
    
    for model_name, results in all_results.items():
        train_metrics = results['train_metrics']
        test_metrics = results['test_metrics']
        
        row = {
            'Model': model_name,
            'Train_Accuracy': train_metrics['Train_accuracy'],
            'Test_Accuracy': test_metrics['Test_accuracy'],
            'Train_F1': train_metrics['Train_f1'],
            'Test_F1': test_metrics['Test_f1'],
            'Train_Precision': train_metrics['Train_precision'],
            'Test_Precision': test_metrics['Test_precision'],
            'Train_Recall': train_metrics['Train_recall'],
            'Test_Recall': test_metrics['Test_recall'],
            'Training_Time': results['training_time']
        }
        
        # ROC-AUC (varsa)
        if 'Train_roc_auc' in train_metrics and train_metrics['Train_roc_auc'] is not None:
            row['Train_ROC_AUC'] = train_metrics['Train_roc_auc']
            row['Test_ROC_AUC'] = test_metrics['Test_roc_auc']
        
        comparison_data.append(row)
    
    # DataFrame oluştur ve sırala
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(metric, ascending=False)
    
    if verbose:
        print(f"\n📊 Model Karşılaştırma Tablosu (sıralama: {metric}):")
        print(f"{'='*100}")
        
        # Temel metrikleri göster
        display_cols = ['Model', 'Test_Accuracy', 'Test_F1', 'Test_Precision', 'Test_Recall', 'Training_Time']
        if 'Test_ROC_AUC' in comparison_df.columns:
            display_cols.insert(-1, 'Test_ROC_AUC')
        
        print(comparison_df[display_cols].round(4).to_string(index=False))
        
        # En iyi model
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 EN İYİ MODEL: {best_model['Model']}")
        print(f"   • Test F1-Score: {best_model['Test_F1']:.4f}")
        print(f"   • Test Accuracy: {best_model['Test_Accuracy']:.4f}")
        print(f"   • Eğitim Süresi: {best_model['Training_Time']:.2f} saniye")
    
    return comparison_df


def save_best_model(all_results: Dict[str, Dict[str, Any]], 
                   comparison_df: pd.DataFrame,
                   model_path: Path, verbose: bool = True) -> str:
    """
    En iyi modeli kaydeder.
    
    Parameters:
        all_results (Dict[str, Dict[str, Any]]): Tüm model sonuçları
        comparison_df (pd.DataFrame): Karşılaştırma tablosu
        model_path (Path): Model kayıt yolu
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        str: En iyi model ismi
    """
    # En iyi modeli seç
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_data = all_results[best_model_name]
    best_model = best_model_data['model']
    
    # Dizin oluştur
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Modeli kaydet
    joblib.dump(best_model, model_path)
    
    if verbose:
        print(f"\n{'='*80}")
        print("EN İYİ MODEL KAYDEDİLİYOR")
        print(f"{'='*80}")
        print(f"🏆 En İyi Model: {best_model_name}")
        print(f"📁 Kayıt Yolu: {model_path}")
        print(f"✅ Model başarıyla kaydedildi!")
    
    return best_model_name


def save_all_models(all_results: Dict[str, Dict[str, Any]], 
                   models_dir: Path, verbose: bool = True) -> None:
    """
    Tüm eğitilmiş modelleri ayrı ayrı kaydeder.
    
    Parameters:
        all_results (Dict[str, Dict[str, Any]]): Tüm model sonuçları
        models_dir (Path): Modellerin kaydedileceği dizin
        verbose (bool): İşlem bilgilerini yazdır
    """
    # Dizin oluştur
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print("TÜM MODELLERİ KAYDETME")
        print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        model = results['model']
        
        # Dosya adını temizle (boşlukları tire yap, küçük harf)
        filename = model_name.lower().replace(' ', '_') + '.pkl'
        filepath = models_dir / filename
        
        # Modeli kaydet
        joblib.dump(model, filepath)
        
        if verbose:
            print(f"✅ {model_name:<25} → {filepath.name}")
    
    if verbose:
        print(f"\n✅ {len(all_results)} model başarıyla kaydedildi!")
        print(f"📁 Kayıt Dizini: {models_dir}")


def load_model(model_path: Path, verbose: bool = True):
    """
    Kaydedilmiş modeli yükler.
    
    Parameters:
        model_path (Path): Model dosya yolu
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        Model: Yüklenen model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    model = joblib.load(model_path)
    
    if verbose:
        print(f"✅ Model başarıyla yüklendi: {model_path}")
    
    return model


def get_model_summary(all_results: Dict[str, Dict[str, Any]], 
                     comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Model eğitimi özeti döner.
    
    Parameters:
        all_results (Dict[str, Dict[str, Any]]): Tüm model sonuçları
        comparison_df (pd.DataFrame): Karşılaştırma tablosu
        
    Returns:
        Dict[str, Any]: Özet bilgiler
    """
    best_model = comparison_df.iloc[0]
    
    summary = {
        'total_models': len(all_results),
        'best_model': best_model['Model'],
        'best_test_f1': best_model['Test_F1'],
        'best_test_accuracy': best_model['Test_Accuracy'],
        'best_training_time': best_model['Training_Time'],
        'model_performance': comparison_df.to_dict('records')
    }
    
    return summary


def print_model_summary(summary: Dict[str, Any]) -> None:
    """
    Model özeti bilgilerini yazdırır.
    
    Parameters:
        summary (Dict[str, Any]): Özet bilgiler
    """
    print(f"\n{'='*80}")
    print("MODEL EĞİTİMİ ÖZETİ")
    print(f"{'='*80}")
    print(f"\n📊 Genel Bilgiler:")
    print(f"   • Toplam Model: {summary['total_models']}")
    print(f"   • En İyi Model: {summary['best_model']}")
    print(f"   • En İyi F1-Score: {summary['best_test_f1']:.4f}")
    print(f"   • En İyi Accuracy: {summary['best_test_accuracy']:.4f}")
    print(f"   • Eğitim Süresi: {summary['best_training_time']:.2f} saniye")
    
    print(f"\n🏆 Model Sıralaması (F1-Score):")
    for i, model in enumerate(summary['model_performance'], 1):
        print(f"   {i}. {model['Model']:<20} F1: {model['Test_F1']:.4f} | Acc: {model['Test_Accuracy']:.4f}")
    
    print(f"{'='*80}\n")