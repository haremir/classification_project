"""
Model Değerlendirme Modülü
==========================

Bu modül, eğitilmiş makine öğrenmesi modellerini detaylı şekilde değerlendirir.

Fonksiyonlar:
    - evaluate_model: Tek bir modeli detaylı değerlendir
    - plot_confusion_matrix: Confusion matrix görselleştir
    - plot_roc_curve: ROC eğrisi çiz
    - plot_feature_importance: Feature importance grafiği
    - plot_precision_recall_curve: Precision-Recall eğrisi
    - generate_classification_report: Sınıflandırma raporu
    - compare_all_models: Tüm modelleri görselleştirerek karşılaştır
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings('ignore')


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_proba: np.ndarray = None,
                   model_name: str = "Model",
                   class_labels: Dict[int, str] = None) -> Dict[str, Any]:
    """
    Modeli detaylı şekilde değerlendirir.
    
    Parameters:
        y_true (np.ndarray): Gerçek değerler
        y_pred (np.ndarray): Tahmin edilen değerler
        y_proba (np.ndarray): Olasılık tahminleri (opsiyonel)
        model_name (str): Model ismi
        class_labels (Dict[int, str]): Sınıf etiketleri
        
    Returns:
        Dict[str, Any]: Değerlendirme sonuçları
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # ROC-AUC (eğer olasılık tahminleri varsa)
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
    else:
        fpr, tpr, thresholds = None, None, None
        roc_auc = None
        precision, recall, pr_thresholds = None, None, None
        avg_precision = None
    
    results = {
        'model_name': model_name,
        'confusion_matrix': cm,
        'classification_report': report,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision_curve': precision,
        'recall_curve': recall,
        'avg_precision': avg_precision
    }
    
    return results


def plot_confusion_matrix(cm: np.ndarray, class_labels: Dict[int, str],
                         model_name: str = "Model",
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Path = None) -> None:
    """
    Confusion matrix'i görselleştirir.
    
    Parameters:
        cm (np.ndarray): Confusion matrix
        class_labels (Dict[int, str]): Sınıf etiketleri
        model_name (str): Model ismi
        figsize (Tuple[int, int]): Grafik boyutu
        save_path (Path): Kayıt yolu (opsiyonel)
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix (yüzdelik)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Sınıf isimleri
    labels = [class_labels[i] for i in sorted(class_labels.keys())]
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Sayı'}, linewidths=1, linecolor='black')
    
    # Her hücreye yüzdelik ekle
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Gerçek Sınıf', fontsize=12, fontweight='bold')
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
                  model_name: str = "Model",
                  figsize: Tuple[int, int] = (8, 6),
                  save_path: Path = None) -> None:
    """
    ROC eğrisini çizer.
    
    Parameters:
        fpr (np.ndarray): False positive rate
        tpr (np.ndarray): True positive rate
        roc_auc (float): ROC-AUC skoru
        model_name (str): Model ismi
        figsize (Tuple[int, int]): Grafik boyutu
        save_path (Path): Kayıt yolu (opsiyonel)
    """
    plt.figure(figsize=figsize)
    
    plt.plot(fpr, tpr, color='#4ECDC4', lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray,
                                avg_precision: float,
                                model_name: str = "Model",
                                figsize: Tuple[int, int] = (8, 6),
                                save_path: Path = None) -> None:
    """
    Precision-Recall eğrisini çizer.
    
    Parameters:
        precision (np.ndarray): Precision değerleri
        recall (np.ndarray): Recall değerleri
        avg_precision (float): Average precision skoru
        model_name (str): Model ismi
        figsize (Tuple[int, int]): Grafik boyutu
        save_path (Path): Kayıt yolu (opsiyonel)
    """
    plt.figure(figsize=figsize)
    
    plt.plot(recall, precision, color='#FF6B6B', lw=3,
             label=f'PR Curve (AP = {avg_precision:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(model, feature_names: List[str],
                           model_name: str = "Model",
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (10, 8),
                           save_path: Path = None) -> pd.DataFrame:
    """
    Feature importance grafiği çizer.
    
    Parameters:
        model: Eğitilmiş model (feature_importances_ özelliği olmalı)
        feature_names (List[str]): Özellik isimleri
        model_name (str): Model ismi
        top_n (int): Gösterilecek en önemli özellik sayısı
        figsize (Tuple[int, int]): Grafik boyutu
        save_path (Path): Kayıt yolu (opsiyonel)
        
    Returns:
        pd.DataFrame: Feature importance tablosu
    """
    # Feature importance al
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression için
        importances = np.abs(model.coef_[0])
    else:
        print(f"⚠️  {model_name} modeli feature importance desteklemiyor.")
        return None
    
    # DataFrame oluştur
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Top N özellik
    top_features = feature_importance_df.head(top_n)
    
    # Görselleştir
    plt.figure(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                    color=colors, edgecolor='black', linewidth=1)
    
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importance - {model_name}', 
              fontsize=14, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return feature_importance_df


def generate_classification_report_df(report_dict: Dict, 
                                      class_labels: Dict[int, str] = None) -> pd.DataFrame:
    """
    Classification report'u DataFrame'e çevirir.
    
    Parameters:
        report_dict (Dict): Classification report (dict formatı)
        class_labels (Dict[int, str]): Sınıf etiketleri
        
    Returns:
        pd.DataFrame: Report tablosu
    """
    # Sadece sınıf bazlı metrikleri al
    report_data = []
    
    for key, metrics in report_dict.items():
        if key in ['-1', '1', -1, 1]:  # Sınıf metrikleri
            class_name = class_labels.get(int(key), str(key)) if class_labels else str(key)
            report_data.append({
                'Class': class_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    # Macro ve weighted avg ekle
    if 'macro avg' in report_dict:
        report_data.append({
            'Class': 'Macro Avg',
            'Precision': report_dict['macro avg']['precision'],
            'Recall': report_dict['macro avg']['recall'],
            'F1-Score': report_dict['macro avg']['f1-score'],
            'Support': report_dict['macro avg']['support']
        })
    
    if 'weighted avg' in report_dict:
        report_data.append({
            'Class': 'Weighted Avg',
            'Precision': report_dict['weighted avg']['precision'],
            'Recall': report_dict['weighted avg']['recall'],
            'F1-Score': report_dict['weighted avg']['f1-score'],
            'Support': report_dict['weighted avg']['support']
        })
    
    return pd.DataFrame(report_data)


def compare_all_models_visual(comparison_df: pd.DataFrame,
                              figsize: Tuple[int, int] = (16, 10),
                              save_path: Path = None) -> None:
    """
    Tüm modelleri görselleştirerek karşılaştırır.
    
    Parameters:
        comparison_df (pd.DataFrame): Model karşılaştırma tablosu
        figsize (Tuple[int, int]): Grafik boyutu
        save_path (Path): Kayıt yolu (opsiyonel)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Test Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.barh(comparison_df['Model'], comparison_df['Test_Accuracy'], 
                     color=['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181'])
    ax1.set_xlabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Karşılaştırması: Test Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax1.axvline(x=0.95, color='red', linestyle='--', linewidth=2, label='Hedef: 95%')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', ha='left', va='center', fontweight='bold')
    
    # 2. Test F1-Score
    ax2 = axes[0, 1]
    bars2 = ax2.barh(comparison_df['Model'], comparison_df['Test_F1'], 
                     color=['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181'])
    ax2.set_xlabel('Test F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model Karşılaştırması: Test F1-Score', fontsize=14, fontweight='bold', pad=15)
    ax2.axvline(x=0.93, color='red', linestyle='--', linewidth=2, label='Hedef: 93%')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', ha='left', va='center', fontweight='bold')
    
    # 3. Precision vs Recall
    ax3 = axes[1, 0]
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    ax3.bar(x_pos - width/2, comparison_df['Test_Precision'], width, 
            label='Precision', color='#4ECDC4', edgecolor='black')
    ax3.bar(x_pos + width/2, comparison_df['Test_Recall'], width, 
            label='Recall', color='#FF6B6B', edgecolor='black')
    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Değer', fontsize=12, fontweight='bold')
    ax3.set_title('Model Karşılaştırması: Precision vs Recall', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax3.axhline(y=0.93, color='red', linestyle='--', linewidth=2, label='Hedef: 93%')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. ROC-AUC (eğer varsa)
    if 'Test_ROC_AUC' in comparison_df.columns:
        ax4 = axes[1, 1]
        bars4 = ax4.barh(comparison_df['Model'], comparison_df['Test_ROC_AUC'], 
                         color=['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181'])
        ax4.set_xlabel('Test ROC-AUC', fontsize=12, fontweight='bold')
        ax4.set_title('Model Karşılaştırması: ROC-AUC', fontsize=14, fontweight='bold', pad=15)
        ax4.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars4):
            width = bar.get_width()
            ax4.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                     f'{width:.4f}', ha='left', va='center', fontweight='bold')
    else:
        # ROC-AUC yoksa Training Time göster
        ax4 = axes[1, 1]
        bars4 = ax4.barh(comparison_df['Model'], comparison_df['Training_Time'], 
                         color=['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181'])
        ax4.set_xlabel('Eğitim Süresi (saniye)', fontsize=12, fontweight='bold')
        ax4.set_title('Model Karşılaştırması: Eğitim Süresi', fontsize=14, fontweight='bold', pad=15)
        ax4.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars4):
            width = bar.get_width()
            ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}s', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def evaluate_all_models(all_results, X_test, y_test, feature_names,
                       class_labels, figures_dir, verbose=True):
    """
    Tüm modelleri tek seferde değerlendirir.
    """
    if verbose:
        print(f"\n{'='*80}")
        print("TÜM MODELLERİN DEĞERLENDİRİLMESİ")
        print(f"{'='*80}")
    
    evaluation_results = {}
    
    for model_name, model_data in all_results.items():
        model = model_data['model']
        
        results, feature_imp = full_model_evaluation(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            model_name=model_name,
            class_labels=class_labels,
            figures_dir=figures_dir,
            verbose=verbose
        )
        
        evaluation_results[model_name] = {
            'results': results,
            'feature_importance': feature_imp
        }
    
    if verbose:
        print(f"\n✅ {len(evaluation_results)} model değerlendirildi!")
    
    return evaluation_results


def full_model_evaluation(model, X_test, y_test, feature_names, 
                         model_name, class_labels, figures_dir, 
                         verbose=True):
    """
    Bir modeli baştan sona değerlendirir ve tüm grafikleri kaydeder.
    
    Bu tek fonksiyon:
    - Tahmin yapar
    - Metrikleri hesaplar
    - Confusion matrix çizer
    - ROC curve çizer
    - Feature importance çizer
    - Precision-Recall curve çizer
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} - TAM DEĞERLENDİRME")
        print(f"{'='*80}")
    
    # 1. Tahminler
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None
    
    # 2. Değerlendirme
    results = evaluate_model(y_test, y_pred, y_proba, model_name, class_labels)
    
    # 3. Classification Report
    if verbose:
        print(f"\n📊 Classification Report:")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred, 
                                   target_names=[class_labels[k] for k in sorted(class_labels.keys())]))
    
    # 4. Confusion Matrix
    plot_confusion_matrix(
        results['confusion_matrix'], 
        class_labels, 
        model_name,
        save_path=figures_dir / f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    )
    
    # 5. ROC Curve (eğer varsa)
    if results['roc_auc'] is not None:
        plot_roc_curve(
            results['fpr'], 
            results['tpr'], 
            results['roc_auc'],
            model_name,
            save_path=figures_dir / f'{model_name.lower().replace(" ", "_")}_roc_curve.png'
        )
        
        # 6. Precision-Recall Curve
        plot_precision_recall_curve(
            results['precision_curve'],
            results['recall_curve'],
            results['avg_precision'],
            model_name,
            save_path=figures_dir / f'{model_name.lower().replace(" ", "_")}_pr_curve.png'
        )
    
    # 7. Feature Importance
    feature_imp_df = plot_feature_importance(
        model,
        feature_names,
        model_name,
        top_n=20,
        save_path=figures_dir / f'{model_name.lower().replace(" ", "_")}_feature_importance.png'
    )
    
    if verbose:
        print(f"\n✅ {model_name} değerlendirmesi tamamlandı!")
    
    return results, feature_imp_df