import pandas as pd
import glob
import os
import sys
import re

# Çıktılarda karakter hatası almamak için
sys.stdout.reconfigure(encoding='utf-8')

def super_clean_name(name):
    """
    Dosya adını en saf ve karşılaştırılabilir formuna dönüştürür.
    Böylece Excel'deki yazım hataları ile Diskteki dosya isimleri eşleşebilir.
    """
    if pd.isna(name): return ""
    name = str(name).lower()
    
    # 1. Türkçe karakterleri İngilizceye çevir
    tr_to_eng = str.maketrans("öüşıçğ", "ousicg")
    name = name.translate(tr_to_eng)
    
    # 2. Uzantıları temizle (.wav, .wav_)
    name = name.replace('.wav_', '').replace('.wav', '')
    
    # 3. İşletim sistemi kopya eklerini (1), (2) vb. temizle
    name = re.sub(r'\s*\(\d+\)', '', name)
    
    # 4. Sadece harf ve rakamları bırak (boşluk, alt tire, tire tamamen silinir)
    name = re.sub(r'[^a-z0-9]', '', name)
    
    return name

def get_real_file_path(base_path, group_folder, raw_excel_name):
    """ Diskteki dosyaları tarar ve süper temizlenmiş isimleriyle eşleştirir. """
    potential_dirs = [
        os.path.join(base_path, group_folder),
        os.path.join(base_path, group_folder.capitalize()),
        os.path.join(base_path, group_folder.upper())
    ]
    
    target_clean = super_clean_name(raw_excel_name)
    if not target_clean:
        return None

    for directory in potential_dirs:
        if os.path.exists(directory):
            try:
                for f in os.listdir(directory):
                    if not f.lower().endswith(('.wav', '.wav_')): 
                        continue # Sadece ses dosyalarını kontrol et
                        
                    disk_clean = super_clean_name(f)
                    
                    if target_clean == disk_clean:
                        return os.path.join(directory, f)
            except OSError:
                continue
    return None

def find_column(df, keywords):
    for col in df.columns:
        if any(kw in str(col).lower() for kw in keywords):
            return col
    return None

# --- ANA KOD ---
base_path = "Dataset"
excel_files = glob.glob(f"{base_path}/**/*.xlsx", recursive=True)
all_data = []

print(f"--- {len(excel_files)} adet Excel dosyası işleniyor... ---")

for file in excel_files:
    try:
        df = pd.read_excel(file)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        col_gender = find_column(df, ['cinsiyet', 'gender'])
        col_age = find_column(df, ['yaş', 'yas', 'age'])
        col_filename = find_column(df, ['dosya', 'file', 'adı', 'adi', 'örnek'])
        
        if not col_filename:
            continue
            
        rename_map = {col_filename: 'file_name'}
        if col_gender: rename_map[col_gender] = 'gender'
        if col_age: rename_map[col_age] = 'age'
        
        df = df.rename(columns=rename_map)
        
        if 'gender' not in df.columns: df['gender'] = 'Bilinmiyor'
        if 'age' not in df.columns: df['age'] = 0
        
        df = df[['gender', 'age', 'file_name']]
        group_folder = os.path.basename(os.path.dirname(file))
        df['source_group'] = group_folder
        
        # Boş satırları at
        df = df.dropna(subset=['file_name'])
        
        # Fiziksel yol eşleştirmesi yap
        df['path'] = df.apply(lambda row: get_real_file_path(base_path, group_folder, row['file_name']), axis=1)
        
        all_data.append(df)
        
    except Exception as e:
        print(f"Okuma Hatası ({file}): {e}")

if all_data:
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Sadece fiziksel olarak yolu bulunan (eşleşmiş) dosyaları tutalım
    matched_df = master_df.dropna(subset=['path'])
    missing_df = master_df[master_df['path'].isnull()]
    
    if not missing_df.empty:
        print(f"\nUYARI: Excel'de yazan ancak diskte fiziksel olarak eşleşmeyen {len(missing_df)} kayıt var.")
    
    matched_df.to_excel("Birlesmis_Metadata.xlsx", index=False)
    print(f"\nİşlem Tamamlandı. Birlesmis_Metadata.xlsx oluşturuldu.")
    print(f"Başarıyla eşleşen ve kaydedilen dosya sayısı: {len(matched_df)} / {len(master_df)}")