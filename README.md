# Phishing Website Detection

## 🎯 Proje Özeti
**Phishing (oltalama) web sitelerini** tespit etmek için supervised machine learning yöntemlerini karşılaştıran proje.

## 📊 Amaç
- Web sitesi özelliklerine göre phishing/legitimate sınıflandırması
- Random Forest, Decision Tree, Gradient Boosting ve Logistic Regression karşılaştırması
- En iyi performans gösteren modeli belirlemek

## 📁 Veri Seti
- **Kaynak**: Phishing Website Detection Dataset (ARFF)
- **Toplam Kayıt**: ~11,000 satır
- **Feature Sayısı**: 30 özellik
- **Label Dağılımı**: Phishing %56, Legitimate %44

## 📈 Ana Bulgular

| Model | Test Acc | F1-Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| **Random Forest** | **96.1%** | **0.961** | **0.989** | **~2s** |
| Gradient Boosting | 96.0% | 0.960 | 0.988 | ~18s |
| Decision Tree | 94.2% | 0.942 | 0.941 | ~0.5s |
| Logistic Regression | 92.1% | 0.921 | 0.972 | ~0.2s |

### 🏆 Kazanan: Random Forest
- Yüksek accuracy (%96.1)
- Düşük overfitting (train-test farkı %1.1)
- Hızlı tahmin süresi
- Feature importance bilgisi

### En Önemli Feature'lar
1. **SSLfinal_State** - SSL sertifikası
2. **URL_of_Anchor** - Anchor URL'leri
3. **Request_URL** - External request oranı
4. **web_traffic** - Trafik sıralaması
5. **Google_Index** - İndekslenme durumu

## 📂 Proje Yapısı
```
phishing-detection/
├── notebooks/           # Jupyter notebooks (CRISP-DM)
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment.ipynb
├── src/                 # Python modülleri
│   ├── config.py
│   ├── data_processing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── automl.py
├── data/
│   ├── raw/
│   └── processed/
├── models/              # Kaydedilmiş modeller
└── reports/
    └── figures/
```

## 📦 Gereksinimler
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter joblib
```

## 💡 Öneriler
1. ✅ **Random Forest'ı production'a deploy et**
2. 📊 **Monitoring sistemi kur** (model drift)
3. 🔄 **Aylık re-training** (yeni phishing teknikleri)
4. 🎯 **Hyperparameter tuning** (GridSearchCV)

## 🔐 Güvenlik Notu
⚠️ **Bu model tek başına yeterli değil!** Ek katmanlar önerilir:
- Google Safe Browsing API
- VirusTotal API
- URL reputation services
- User reporting

## 📅 Proje Bilgileri
- **Metodoloji**: CRISP-DM
- **Toplam Süre**: 2 Gün
- **Model Sayısı**: 4
- **En İyi Sonuç**: %96.1 accuracy

## 📄 Lisans
Bu proje eğitim amaçlıdır.
