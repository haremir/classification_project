"""
Veri İşleme Modülü
==================

Bu modül, phishing tespit projesi için veri yükleme, temizleme,
dönüştürme ve hazırlama fonksiyonlarını içerir.

Fonksiyonlar:
    - load_arff_data: ARFF dosyasını yükle
    - remove_duplicates: Tekrarlı satırları kaldır
    - find_high_correlation_features: Yüksek korelasyonlu özellik çiftlerini bul
    - remove_multicollinear_features: Multicollinear özellikleri çıkar
    - split_data: Train-test split yap
    - save_processed_data: İşlenmiş veriyi kaydet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Set, Dict
from scipy.io import arff
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def load_arff_data(file_path: Path) -> pd.DataFrame:
    """
    ARFF formatındaki veri dosyasını yükler ve pandas DataFrame'e çevirir.
    
    Parameters:
        file_path (Path): ARFF dosyasının yolu
        
    Returns:
        pd.DataFrame: Yüklenmiş veri seti
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        Exception: Yükleme hatası
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
    
    try:
        # ARFF dosyasını yükle
        data_arff, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data_arff)
        
        # Byte string'leri decode et
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except:
                    pass
        
        # Veri tiplerini integer'a çevir
        df = df.astype(int)
        
        print(f"✅ ARFF dosyası başarıyla yüklendi: {file_path.name}")
        print(f"   Boyut: {df.shape[0]:,} satır x {df.shape[1]} sütun")
        
        return df
        
    except Exception as e:
        print(f"❌ Yükleme hatası: {e}")
        raise


def remove_duplicates(df: pd.DataFrame, keep: str = 'first', 
                     verbose: bool = True) -> pd.DataFrame:
    """
    Veri setindeki tekrarlı satırları kaldırır.
    
    Parameters:
        df (pd.DataFrame): Veri seti
        keep (str): Hangi tekrarı tutacağız ('first', 'last', False)
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        pd.DataFrame: Temizlenmiş veri seti
    """
    original_shape = df.shape[0]
    duplicate_count = df.duplicated().sum()
    
    if verbose:
        print(f"\n{'='*80}")
        print("TEKRARLI SATIRLARI TEMİZLEME")
        print(f"{'='*80}")
        print(f"\n📊 Önceki Durum:")
        print(f"   • Toplam Satır: {original_shape:,}")
        print(f"   • Tekrarlı Satır: {duplicate_count:,}")
        print(f"   • Benzersiz Satır: {original_shape - duplicate_count:,}")
    
    # Tekrarları kaldır
    df_clean = df.drop_duplicates(keep=keep)
    new_shape = df_clean.shape[0]
    removed_count = original_shape - new_shape
    
    if verbose:
        print(f"\n📊 Sonraki Durum:")
        print(f"   • Toplam Satır: {new_shape:,}")
        print(f"   • Kaldırılan Satır: {removed_count:,}")
        print(f"   • Veri Kaybı: {(removed_count / original_shape * 100):.2f}%")
        print(f"\n✅ Temizleme tamamlandı!")
    
    return df_clean


def find_high_correlation_features(df: pd.DataFrame, 
                                   feature_names: List[str],
                                   target_name: str,
                                   threshold: float = 0.8,
                                   verbose: bool = True) -> Set[str]:
    """
    Yüksek korelasyonlu özellik çiftlerini bulur ve hedef değişkenle
    korelasyonu düşük olanları çıkarılacak liste olarak döner.
    
    Parameters:
        df (pd.DataFrame): Veri seti
        feature_names (List[str]): Özellik isimleri listesi
        target_name (str): Hedef değişken ismi
        threshold (float): Korelasyon eşiği (varsayılan: 0.8)
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        Set[str]: Çıkarılacak özellikler kümesi
    """
    # Sadece özelliklerin korelasyon matrisi
    X = df[feature_names]
    corr_matrix = X.corr()
    
    # Yüksek korelasyonlu çiftleri bul
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr_pairs.append({
                    'Özellik 1': corr_matrix.columns[i],
                    'Özellik 2': corr_matrix.columns[j],
                    'Korelasyon': corr_matrix.iloc[i, j]
                })
    
    if verbose:
        print(f"\n{'='*80}")
        print("YÜKSEK KORELASYONLU ÖZELLİKLERİN TESPİTİ")
        print(f"{'='*80}")
    
    if len(high_corr_pairs) == 0:
        if verbose:
            print(f"\n✅ {threshold} eşiğinde yüksek korelasyonlu özellik çifti bulunamadı!")
        return set()
    
    if verbose:
        print(f"\n⚠️  {len(high_corr_pairs)} yüksek korelasyonlu çift tespit edildi (|r| ≥ {threshold}):")
        for pair in high_corr_pairs:
            print(f"   • {pair['Özellik 1']} ↔ {pair['Özellik 2']}: {pair['Korelasyon']:.4f}")
    
    # Hedef değişkenle korelasyonları hesapla
    target_corr = df[feature_names + [target_name]].corr()[target_name].drop(target_name)
    
    # Her çiftten hedef ile korelasyonu düşük olanı seç
    features_to_drop = set()
    
    if verbose:
        print(f"\n📋 Çıkarılacak Özelliklerin Belirlenmesi:")
        print("   (Her çiftten hedef değişkenle korelasyonu düşük olanı çıkarılacak)")
        print(f"\n   {'Özellik 1':<30} {'r(target)':<12} {'Özellik 2':<30} {'r(target)':<12} {'Çıkarılacak':<30}")
        print(f"   {'-'*115}")
    
    for pair in high_corr_pairs:
        feat1 = pair['Özellik 1']
        feat2 = pair['Özellik 2']
        corr1 = abs(target_corr[feat1])
        corr2 = abs(target_corr[feat2])
        
        # Hedef ile korelasyonu düşük olanı çıkar
        to_drop = feat1 if corr1 < corr2 else feat2
        features_to_drop.add(to_drop)
        
        if verbose:
            print(f"   {feat1:<30} {corr1:<12.4f} {feat2:<30} {corr2:<12.4f} {to_drop:<30}")
    
    if verbose:
        print(f"\n✅ Çıkarılacak Özellikler ({len(features_to_drop)} adet):")
        for feat in sorted(features_to_drop):
            print(f"   • {feat} (hedef korelasyon: {abs(target_corr[feat]):.4f})")
    
    return features_to_drop


def remove_multicollinear_features(df: pd.DataFrame,
                                   feature_names: List[str],
                                   target_name: str,
                                   threshold: float = 0.8,
                                   verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Multicollinear özellikleri tespit edip çıkarır.
    
    Parameters:
        df (pd.DataFrame): Veri seti
        feature_names (List[str]): Özellik isimleri listesi
        target_name (str): Hedef değişken ismi
        threshold (float): Korelasyon eşiği
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (Temizlenmiş veri, kalan özellikler listesi)
    """
    # Çıkarılacak özellikleri bul
    features_to_drop = find_high_correlation_features(
        df, feature_names, target_name, threshold, verbose
    )
    
    # Kalan özellikler
    remaining_features = [feat for feat in feature_names if feat not in features_to_drop]
    
    # Veri setini güncelle
    df_cleaned = df[remaining_features + [target_name]].copy()
    
    if verbose:
        print(f"\n{'='*80}")
        print("ÖZET")
        print(f"{'='*80}")
        print(f"   • Başlangıç Özellik Sayısı: {len(feature_names)}")
        print(f"   • Çıkarılan Özellik: {len(features_to_drop)}")
        print(f"   • Kalan Özellik: {len(remaining_features)}")
        print(f"\n✅ Multicollinearity temizleme tamamlandı!")
    
    return df_cleaned, remaining_features


def split_data(df: pd.DataFrame,
              feature_names: List[str],
              target_name: str,
              test_size: float = 0.2,
              random_state: int = 42,
              stratify: bool = True,
              verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                             pd.Series, pd.Series]:
    """
    Veri setini train ve test olarak böler.
    
    Parameters:
        df (pd.DataFrame): Veri seti
        feature_names (List[str]): Özellik isimleri listesi
        target_name (str): Hedef değişken ismi
        test_size (float): Test seti oranı (varsayılan: 0.2)
        random_state (int): Random seed (tekrarlanabilirlik için)
        stratify (bool): Sınıf dağılımını koru
        verbose (bool): İşlem bilgilerini yazdır
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X = df[feature_names].copy()
    y = df[target_name].copy()
    
    if verbose:
        print(f"\n{'='*80}")
        print("VERİ BÖLME (TRAIN-TEST SPLIT)")
        print(f"{'='*80}")
        print(f"\n📊 Veri Seti Bilgileri:")
        print(f"   • Toplam Veri: {len(X):,} satır")
        print(f"   • Özellik Sayısı: {X.shape[1]}")
        print(f"   • Hedef Değişken: {target_name}")
        print(f"\n⚙️  Split Parametreleri:")
        print(f"   • Train: %{(1-test_size)*100:.0f}")
        print(f"   • Test: %{test_size*100:.0f}")
        print(f"   • Random State: {random_state}")
        print(f"   • Stratified: {'Evet' if stratify else 'Hayır'}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    if verbose:
        print(f"\n✅ Split tamamlandı!")
        print(f"\n📊 Train Set:")
        print(f"   • Boyut: {X_train.shape[0]:,} satır x {X_train.shape[1]} sütun")
        train_class_dist = y_train.value_counts().sort_index()
        for cls, count in train_class_dist.items():
            percentage = (count / len(y_train)) * 100
            print(f"   • Sınıf {cls:2}: {count:,} ({percentage:.2f}%)")
        
        print(f"\n📊 Test Set:")
        print(f"   • Boyut: {X_test.shape[0]:,} satır x {X_test.shape[1]} sütun")
        test_class_dist = y_test.value_counts().sort_index()
        for cls, count in test_class_dist.items():
            percentage = (count / len(y_test)) * 100
            print(f"   • Sınıf {cls:2}: {count:,} ({percentage:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       feature_names: List[str],
                       target_name: str,
                       train_path: Path,
                       test_path: Path,
                       feature_path: Path = None,
                       verbose: bool = True) -> None:
    """
    İşlenmiş train ve test verilerini diske kaydeder.
    
    Parameters:
        X_train (pd.DataFrame): Train özellikleri
        X_test (pd.DataFrame): Test özellikleri
        y_train (pd.Series): Train hedef değişken
        y_test (pd.Series): Test hedef değişken
        feature_names (List[str]): Özellik isimleri listesi
        target_name (str): Hedef değişken ismi
        train_path (Path): Train dosyası kayıt yolu
        test_path (Path): Test dosyası kayıt yolu
        feature_path (Path): Özellik listesi kayıt yolu (opsiyonel)
        verbose (bool): İşlem bilgilerini yazdır
    """
    if verbose:
        print(f"\n{'='*80}")
        print("İŞLENMİŞ VERİYİ KAYDETME")
        print(f"{'='*80}")
    
    # Dizinleri oluştur
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Train ve test'i birleştir (hedef dahil)
    train_out = pd.concat([X_train, y_train.rename(target_name)], axis=1)
    test_out = pd.concat([X_test, y_test.rename(target_name)], axis=1)
    
    # Kaydet
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    
    if verbose:
        print(f"\n✅ Veri dosyaları kaydedildi:")
        print(f"   • Train: {train_path}")
        print(f"   • Test:  {test_path}")
        print(f"\n📊 Dosya Boyutları:")
        print(f"   • Train: {train_out.shape[0]:,} satır x {train_out.shape[1]} sütun")
        print(f"   • Test:  {test_out.shape[0]:,} satır x {test_out.shape[1]} sütun")
    
    # Özellik listesini kaydet (opsiyonel)
    if feature_path is not None:
        with open(feature_path, 'w', encoding='utf-8') as f:
            for feat in feature_names:
                f.write(feat + '\n')
        if verbose:
            print(f"   • Özellikler: {feature_path}")
            print(f"   • Özellik Sayısı: {len(feature_names)}")
    
    if verbose:
        print(f"\n✅ Tüm dosyalar başarıyla kaydedildi!")


def get_data_summary(df: pd.DataFrame, 
                    feature_names: List[str],
                    target_name: str) -> Dict:
    """
    Veri seti hakkında özet bilgi döner.
    
    Parameters:
        df (pd.DataFrame): Veri seti
        feature_names (List[str]): Özellik isimleri
        target_name (str): Hedef değişken ismi
        
    Returns:
        Dict: Özet bilgiler
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'feature_count': len(feature_names),
        'target_name': target_name,
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'class_distribution': df[target_name].value_counts().to_dict(),
        'class_balance_ratio': df[target_name].value_counts().max() / df[target_name].value_counts().min()
    }
    return summary


def print_data_summary(summary: Dict, verbose: bool = True) -> None:
    """
    Veri özeti bilgilerini yazdırır.
    
    Parameters:
        summary (Dict): Özet bilgiler
        verbose (bool): Detaylı yazdırma
    """
    if not verbose:
        return
    
    print(f"\n{'='*80}")
    print("VERİ SETİ ÖZETİ")
    print(f"{'='*80}")
    print(f"\n📊 Temel Bilgiler:")
    print(f"   • Toplam Satır: {summary['total_rows']:,}")
    print(f"   • Toplam Sütun: {summary['total_columns']}")
    print(f"   • Özellik Sayısı: {summary['feature_count']}")
    print(f"   • Hedef Değişken: {summary['target_name']}")
    
    print(f"\n🔍 Veri Kalitesi:")
    print(f"   • Eksik Değer: {summary['missing_values']:,}")
    print(f"   • Tekrarlı Satır: {summary['duplicate_rows']:,}")
    print(f"   • Bellek Kullanımı: {summary['memory_usage_mb']:.2f} MB")
    
    print(f"\n📈 Sınıf Dağılımı:")
    for cls, count in sorted(summary['class_distribution'].items()):
        percentage = (count / summary['total_rows']) * 100
        print(f"   • Sınıf {cls:2}: {count:,} ({percentage:.2f}%)")
    
    print(f"\n⚖️  Dengesizlik Oranı: {summary['class_balance_ratio']:.2f}:1")
    print(f"{'='*80}\n")