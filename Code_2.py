# ---------------------------------------------------------
# Ses İşareti (Voice Signal) Analiz ve Cinsiyet Sınıflandırma
# - kural tabanlı F0 eşiklerine dayalı temel sınıflandırma
# - RandomForest tabanlı ML eğitimi + çapraz doğrulama
# - Streamlit arayüzüyle tekil ve toplu analitik mod
# ---------------------------------------------------------

import os
import sys
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
import warnings

# librosa uyarılarını gizlemek için
warnings.filterwarnings('ignore')

# Windows dışında veya yerel ortamda Türkçe çıktı garantisi
sys.stdout.reconfigure(encoding='utf-8')

def classify_gender_rule_based(f0):
    """
    Talimatnamede istenen asıl kural tabanlı (Rule-Based) sınıflandırıcı.
    Bu fonksiyon projenin temel gereksinimidir, kaldırılamaz.

    f0 (Hz) değerine göre sınıflandırma yapar:
      - <=0: Bilinmiyor
      - <145: Erkek (E)
      - <225: Kadın (K)
      - >=225: Çocuk (C)
    """
    if f0 <= 0:
        return 'Bilinmiyor'  # Geçersiz / ses yok
    if f0 < 145:
        return 'E'  # W erkek tipik aralığı
    elif f0 < 225:  
        return 'K'  # Kadın aralığı
    else:
        return 'C'  # Çocuk aralığı

def get_f0_via_autocorr(y, sr, frame_length, hop_length, min_f=65, max_f=450):
    """
    Sinyal üzerinde pencereler halinde otokorelasyon (R_tau) ile F0 hesaplar.
    - Girdiler: ham waveform y, örnekleme hızı sr, pencere/hop boyutları.
    - min_f / max_f: insan sesi için güvenilir frekans aralığı.
    - Çıkış: öntanımlı olarak medyan F0 (Hz).
    """
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    f0_list = []
    
    # Arama limiti gecikme değerleri (lag)
    low_lag = int(sr / max_f)
    high_lag = int(sr / min_f)

    # Sesli (Voiced) bölge tespiti: düşük enerjili sessiz bölge atlanır
    energies = np.sum(frames**2, axis=0)
    e_thresh = np.mean(energies) * 0.4 

    for i in range(frames.shape[1]):
        if energies[i] < e_thresh:
            continue
            
        frame = frames[:, i]
        r = librosa.autocorrelate(frame)
        
        # Aranan lag aralığında en yüksek tepeyi bul
        if len(r) > low_lag:
            search_range = r[low_lag:min(high_lag, len(r))]
            if len(search_range) > 0:
                peak = np.argmax(search_range) + low_lag
                f0 = sr / peak
                # Geçerli F0 aralığında ise ekle
                if min_f <= f0 <= max_f:
                    f0_list.append(f0)
                
    return np.median(f0_list) if f0_list else 0  # Sesli bölge yoksa 0 döndür

def analyze_audio_features(file_path):
    """
    Zaman düzlemi özniteliklerini (F0, ZCR, Enerji) ve 
    kıyaslama amaçlı ML özniteliklerini (MFCC, Centroid) çıkarır.

    return: sözlük biçiminde hesaplanan tüm özellikler + grafik verileri.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)  # orijinal örnekleme hızında yükle
        
        # 25 ms pencereleme / 10 ms kayma
        frame_length = int(0.025 * sr) 
        hop_length = int(0.010 * sr)
        
        # 1. Temel Sinyal İşleme Öznitelikleri (Talimatname Gereksinimi)
        avg_f0 = get_f0_via_autocorr(y, sr, frame_length, hop_length)
        zcr_series = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        avg_zcr = np.mean(zcr_series) * sr  # ZCR tek birim sürede eşik geçiş sayısı
        energy_series = np.array([np.sum(x**2) for x in librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T])
        avg_energy = np.mean(energy_series)
        
        # 2. Makine Öğrenmesi İçin Gelişmiş Öznitelikler (Ekstra/Bonus)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)  # her MFCC bileşeni için ortalama
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Özellik vektörü (ML modeli için)
        ml_vector = np.hstack(([avg_f0, avg_zcr, avg_energy, centroid], mfccs_mean))
        
        # Grafik için örnek veri (sinyalin ortasındaki pencereden)
        mid = len(y) // 2
        sample_frame = y[mid : mid + frame_length]
        r_sample = librosa.autocorrelate(sample_frame)
        fft_sample = np.abs(np.fft.rfft(sample_frame * np.hanning(len(sample_frame))))
        fft_freqs = np.fft.rfftfreq(len(sample_frame), 1/sr)

        return {
            'avg_f0': avg_f0,
            'avg_zcr': avg_zcr,
            'avg_energy': avg_energy,
            'ml_vector': ml_vector,
            'r_sample': r_sample,
            'fft_sample': fft_sample,
            'fft_freqs': fft_freqs,
            'sr': sr,
            'signal': y
        }
    except Exception as e:
        st.error(f"Dosya işleme hatası: {e}")
        return None

def batch_analysis_and_train(master_df):
    """
    Toplu veriyi okur, Random Forest modelini eğitir ve başarıyı ölçer.
    - master_df içinde path+gender bulunan satırlar kullanılır.
    - Analiz sonuçları Streamlit tablosu + confusion matrix olarak sunulur.
    """
    st.info(f"Toplam {len(master_df)} dosya analiz ediliyor ve Model Eğitiliyor. Lütfen bekleyin...")
    progress = st.progress(0)
    
    X = []
    y_true = []
    file_records = []
    
    # Veri Çıkarımı
    for i, (idx, row) in enumerate(master_df.iterrows()):
        path = row['path']
        if pd.notna(path) and os.path.exists(str(path)):
            features = analyze_audio_features(path)
            if features:
                label = str(row['gender']).strip().upper()
                if label in ['E', 'K', 'C']:
                    X.append(features['ml_vector'])
                    y_true.append(label)
                    
                    # CSV/Excel raporlama için kaydet
                    file_records.append({
                        'Dosya': os.path.basename(path),
                        'Gerçek': label,
                        'F0': features['avg_f0'],
                        'ZCR': features['avg_zcr'],
                        'Enerji': features['avg_energy']
                    })
        # Her döngüde ilerleme çubuğunu güncelle
        progress.progress((i + 1) / len(master_df))
    
    # Yeterli örnek yoksa erken çık
    if len(X) < 10:
        st.error("Modeli eğitebilmek için yeterli veri bulunamadı (En az 10 geçerli dosya gerekli).")
        return

    # Veri Ölçeklendirme (özelliklerin standardizasyonu)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Kurulumu ve Çapraz Doğrulama
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    st.write("Aşırı öğrenmeyi (overfitting) önlemek için 5-Katlı Çapraz Doğrulama (Cross-Validation) yapılıyor...")
    cv_predictions = cross_val_predict(model, X_scaled, y_true, cv=5)
    
    # Genel Eğitimi Tamamlayıp Oturuma Kaydet
    model.fit(X_scaled, y_true)
    st.session_state.model = model
    st.session_state.scaler = scaler
    
    # Çapraz doğrulama tahminlerini kayıt tablosuna ekle
    for i, record in enumerate(file_records):
        record['Tahmin (ML)'] = cv_predictions[i]
        
    res_df = pd.DataFrame(file_records)

    
    
    st.success("Model Başarıyla Eğitildi! Artık 'Tekil Analiz' sekmesinde Makine Öğrenmesi sonuçlarını da görebilirsiniz.")
    
    st.subheader("📊 Sınıf Bazlı İstatistiksel Bulgular ve ML Başarısı")
    stats = res_df.groupby('Gerçek').agg(
        Örnek_Sayısı=('F0', 'count'),
        Ortalama_F0=('F0', 'mean'),
        Standart_Sapma_F0=('F0', 'std')
    ).reset_index()
    
    # Her gerçek label için doğruluk yüzdesini hesapla
    success_list = []
    for label in stats['Gerçek']:
        sub = res_df[res_df['Gerçek'] == label]
        acc = accuracy_score(sub['Gerçek'], sub['Tahmin (ML)']) * 100
        success_list.append(f"%{acc:.2f}")
    stats['Model Başarısı'] = success_list
    st.table(stats)
    
    st.subheader("📉 Karışıklık Matrisi (Confusion Matrix)")
    cm = confusion_matrix(res_df['Gerçek'], res_df['Tahmin (ML)'], labels=['E', 'K', 'C'])
    cm_display = pd.DataFrame(cm, index=['Gerçek E', 'Gerçek K', 'Gerçek C'], columns=['Tahmin E', 'Tahmin K', 'Tahmin C'])
    st.dataframe(cm_display.style.background_gradient(cmap='Blues'))
    
    # Hatalı tahminleri filtrele ve Excel'e kaydet
    hatali_df = res_df[res_df['Gerçek'] != res_df['Tahmin (ML)']]
    if not hatali_df.empty:
        hatali_df.to_excel('Hatali_Tahminler.xlsx', index=False)
        st.success(f"Hatalı tahminler 'Hatali_Tahminler.xlsx' dosyasına kaydedildi. Toplam {len(hatali_df)} hatalı tahmin.")
    else:
        st.info("Hiç hatalı tahmin bulunmadı.")

def main():
    # Sayfa yapılandırması, başlık ve durum sıfırlama
    st.set_page_config(page_title="Ses Sınıflandırma Sistemi", layout="wide")
    st.title("🎙️ Ses İşareti Analizi ve Cinsiyet Sınıflandırma")
    
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.scaler = None

    # İki sekme: gerçek zamanlı tekli analiz + toplu model eğitimi
    tab1, tab2 = st.tabs(["🎯 Tekil Analiz (Canlı Demo)", "📊 Toplu Eğitim ve Analiz"])
    
    with tab1:
        st.header("Zaman ve Frekans Düzlemi Analizi")
        st.write("Burada bir ses dosyası yükleyerek **Kural Tabanlı (Rule-Based)** ve eğer model eğitildiyse **Makine Öğrenmesi** sınıflandırmasını anında görebilirsiniz.")
        
        uploaded = st.file_uploader("Test etmek için bir .wav dosyası seçin", type="wav")
        if uploaded:
            temp_path = "temp_analysis.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            
            feat = analyze_audio_features(temp_path)
            if feat:
                st.audio(temp_path)
                
                # Temel Kural Tabanlı Sınıflandırma (Her zaman çalışır)
                rule_based_pred = classify_gender_rule_based(feat['avg_f0'])
                
                # Makine Öğrenmesi Sınıflandırması (Sadece model eğitildiyse çalışır)
                ml_pred_text = "Eğitilmedi"
                ml_conf_text = "-"
                
                if st.session_state.model is not None:
                    ml_vector_scaled = st.session_state.scaler.transform([feat['ml_vector']])
                    ml_pred = st.session_state.model.predict(ml_vector_scaled)[0]
                    ml_probs = st.session_state.model.predict_proba(ml_vector_scaled)[0]
                    ml_pred_text = str(ml_pred)
                    ml_conf_text = f"%{max(ml_probs)*100:.1f}"

                # Arayüzde sonuç kartları gösterimi
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("F0 (Otokorelasyon)", f"{feat['avg_f0']:.2f} Hz")
                c2.metric("Kural Tabanlı Tahmin", rule_based_pred)
                c3.metric("ML Sınıf Tahmini", ml_pred_text)
                c4.metric("ML Tahmin Güveni", ml_conf_text)
                
                if st.session_state.model is None:
                    st.warning("💡 İpucu: Makine öğrenmesi sonuçlarını görmek için 'Toplu Eğitim ve Analiz' sekmesinden modeli eğitebilirsiniz.")

                # Grafikler: Otokorelasyon + FFT spektrumu
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                ax1.plot(feat['r_sample'], color='#1f77b4')
                ax1.set_title("Otokorelasyon Fonksiyonu R(τ)")
                ax1.set_xlabel("Gecikme (Lag)")
                ax1.grid(alpha=0.2)
                
                ax2.plot(feat['fft_freqs'], feat['fft_sample'], color='#2ca02c')
                ax2.set_xlim(0, 1000)
                ax2.set_title("FFT Büyüklük Spektrumu")
                ax2.set_xlabel("Frekans (Hz)")
                ax2.grid(alpha=0.2)
                
                if feat['avg_f0'] > 0:
                    ax2.axvline(x=feat['avg_f0'], color='red', linestyle='--', label=f"Tespit Edilen F0: {feat['avg_f0']:.1f}Hz")
                    ax2.legend()

                st.pyplot(fig)

    with tab2:
        st.header("Veri Seti Performans Raporu ve Model Eğitimi")
        st.markdown("""
        **Neden Makine Öğrenmesi (Bonus)?**
        Kural tabanlı (F0 eşiklerine dayalı) sınıflandırma projenin ana gereksinimi olsa da, kadın ve çocuk frekanslarının bazı durumlarda örtüşmesi sebebiyle Random Forest modeli (MFCC öznitelikleri ile birlikte) eğitilerek projenin genelleştirme yeteneği artırılmıştır.
        """)

        # Veri kaynağı mevcut mu kontrolü
        if os.path.exists('Birlesmis_Metadata.xlsx'):
            df = pd.read_excel('Birlesmis_Metadata.xlsx')
            # Kullanıcı düğmeye basarsa toplu model eğitimi başlat
            if st.button("Toplu Analizi ve Eğitimi Başlat"):
                batch_analysis_and_train(df)
        else:
            # Excel dosyası yoksa hata göster
            st.error("'Birlesmis_Metadata.xlsx' bulunamadı. Lütfen klasörünüzde bu dosyanın olduğundan emin olun.")

# Script doğrudan çalıştırılmışsa uygulamayı başlat
if __name__ == "__main__":
    main()