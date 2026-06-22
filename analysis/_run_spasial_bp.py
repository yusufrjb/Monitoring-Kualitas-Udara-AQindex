# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path.cwd().parent
DATA_SPASIAL = BASE / 'data_spasial'
FEATURES = ['pm25_ugm3', 'pm10_ugm3', 'co_ugm3', 'no2_ugm3', 'o3_ugm3']

# Breakpoints ISPU per PermenLHK 14/2020 (ug/m3)
BP_PM25 = [(0,15.5,0,50),(15.5,55.4,50,100),(55.4,150.4,100,200),(150.4,250.4,200,300),(250.4,500,300,500)]
BP_PM10 = [(0,50,0,50),(50,150,50,100),(150,350,100,200),(350,420,200,300),(420,500,300,500)]
BP_CO   = [(0,4000,0,50),(4000,8000,50,100),(8000,15000,100,200),(15000,30000,200,300),(30000,45000,300,500)]
BP_NO2  = [(0,80,0,50),(80,200,50,100),(200,1130,100,200),(1130,2260,200,300),(2260,3000,300,500)]
BP_O3   = [(0,120,0,50),(120,235,50,100),(235,400,100,200),(400,800,200,300),(800,1000,300,500)]

def conc_to_ispi(val, bp):
    if val <= 0: return 0
    for cl, ch, il, ih in bp:
        if val <= ch: return il + (val - cl) / (ch - cl) * (ih - il)
    return bp[-1][3]

def ispi_to_label(ispi):
    if ispi <= 50: return 'Baik'
    if ispi <= 100: return 'Sedang'
    if ispi <= 200: return 'Tidak Sehat'
    if ispi <= 300: return 'Sangat Tidak Sehat'
    return 'Berbahaya'

def label_from_row(row):
    pm25, pm10, co, no2, o3 = row['pm25_ugm3'], row['pm10_ugm3'], row['co_ugm3'], row['no2_ugm3'], row['o3_ugm3']
    ispis = [conc_to_ispi(pm25, BP_PM25), conc_to_ispi(pm10, BP_PM10),
             conc_to_ispi(co, BP_CO), conc_to_ispi(no2, BP_NO2),
             conc_to_ispi(o3, BP_O3)]
    return ispi_to_label(max(ispis))


print("Menggunakan ISPU Breakpoint (PermenLHK 14/2020)")
print("Kategori: Baik, Sedang, Tidak Sehat, Sangat Tidak Sehat, Berbahaya")

df = pd.read_csv(DATA_SPASIAL / 'tb_konsentrasi_gas_rows.csv', parse_dates=['created_at'])
print(f'Total data: {len(df)} baris')
print(f'Rentang: {df["created_at"].iloc[0]} - {df["created_at"].iloc[-1]}')

# Deteksi gap >1 jam (3600 detik)
time_diffs = df['created_at'].diff().dt.total_seconds()
gap_rows = time_diffs[time_diffs > 3600].index.tolist()

print(f'Gap >1 jam ditemukan: {len(gap_rows)}')
for idx in gap_rows:
    gap_min = time_diffs[idx] / 60
    print(f'  Baris {idx}: {gap_min:.0f} menit')
    print(f'    Sebelum: {df["created_at"].iloc[idx-1]}')
    print(f'    Sesudah: {df["created_at"].iloc[idx]}')

# Beri label segmen berdasarkan gap
seg = 0
segmen_labels = []
for i in range(len(df)):
    if i in gap_rows:
        seg += 1
    segmen_labels.append(seg)

df['segmen'] = segmen_labels
df['segmen'] = df['segmen'].replace(0, 1)  # gabung segmen 0 -> 1 (hanya 1 baris)

# Ambil semua data per segmen
segments = []
for s in sorted(df['segmen'].unique()):
    sub = df[df['segmen'] == s].copy()
    segments.append(sub)
df_seg = pd.concat(segments).sort_values('created_at').reset_index(drop=True)

LOKASI_MAP = {
    1: 'Ngagel (Pagi - Urban Traffic)',
    2: 'Rungkut (Siang - Industri Manufaktur)',
    3: 'Margomulyo (Sore - Industri Logistik)'
}
df_seg['lokasi'] = df_seg['segmen'].map(LOKASI_MAP)

for s in sorted(df_seg['segmen'].unique()):
    sub = df_seg[df_seg['segmen'] == s]
    print(f'{LOKASI_MAP[s]}: {len(sub)} baris, '
          f'{sub["created_at"].iloc[0]} - {sub["created_at"].iloc[-1]}')

df_seg['label_bp'] = df_seg.apply(label_from_row, axis=1)

for s in sorted(df_seg['segmen'].unique()):
    sub = df_seg[df_seg['segmen'] == s]
    print(f'\n{"="*60}')
    print(f'Wilayah: {LOKASI_MAP[s]}')
    print(f'{"="*60}')
    print(sub[['created_at', 'pm25_ugm3', 'pm10_ugm3', 'co_ugm3', 'label_bp']].to_string(index=False))

# Simpan hasil
df_seg.to_csv(BASE / 'analysis' / 'hasil_klasifikasi_spasial_bp.csv', index=False)
print(f"\nSaved: hasil_klasifikasi_spasial_bp.csv ({len(df_seg)} rows)")

# Load sampled dataset and show summary table with percentages
import pandas as pd
from pathlib import Path
BASE = Path.cwd().parent
sampled = pd.read_csv(BASE / 'analysis' / 'sampled_60_per_wilayah.csv', parse_dates=['created_at'])

summary = sampled.groupby(['lokasi', 'predicted_category']).size().reset_index(name='jumlah')
summary['persentase'] = summary.groupby('lokasi')['jumlah'].transform(lambda x: (x / x.sum() * 100).round(1).astype(str) + '%')
summary = summary.sort_values(['lokasi', 'predicted_category']).reset_index(drop=True)
print(summary.to_string(index=False))

# Save summary CSV
summary.to_csv(BASE / 'analysis' / 'sampled_summary_by_lokasi.csv', index=False)
print('\nWrote summary to sampled_summary_by_lokasi.csv')

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.cwd().parent
sampled = pd.read_csv(BASE / 'analysis' / 'sampled_60_per_wilayah.csv', parse_dates=['created_at'])

POLUTAN = ['pm25_ugm3', 'pm10_ugm3', 'co_ugm3', 'no2_ugm3', 'o3_ugm3']
POLUTAN_LABEL = {'pm25_ugm3': 'PM2.5', 'pm10_ugm3': 'PM10', 'co_ugm3': 'CO', 'no2_ugm3': 'NO2', 'o3_ugm3': 'O3'}
LOKASI_ORDER = sampled['lokasi'].unique()

results = []

for pol in POLUTAN:
    groups = [sampled[sampled['lokasi'] == l][pol].dropna().values for l in LOKASI_ORDER]
    
    # Shapiro-Wilk normality test for each group
    sw_results = []
    for l, g in zip(LOKASI_ORDER, groups):
        if len(g) >= 3:
            stat_sw, p_sw = stats.shapiro(g)
            sw_results.append(f"{l}: stat={stat_sw:.4f}, p={p_sw:.4f} {'Normal' if p_sw > 0.05 else 'Tidak Normal'}")
        else:
            sw_results.append(f"{l}: insufficient data")
    
    # Levene's test for homogeneity of variances
    stat_lev, p_lev = stats.levene(*groups)
    
    # One-way ANOVA
    f_stat, p_anova = stats.f_oneway(*groups)
    
    # Post-hoc Tukey if ANOVA significant
    tukey = None
    if p_anova < 0.05:
        tukey = pairwise_tukeyhsd(sampled[pol], sampled['lokasi'], alpha=0.05)
    
    results.append({
        'polutan': pol,
        'shapiro': sw_results,
        'levene_stat': stat_lev,
        'levene_p': p_lev,
        'f_stat': f_stat,
        'p_anova': p_anova,
        'tukey': tukey,
    })
    
    print(f"\n{'='*60}")
    print(f"ANOVA — {POLUTAN_LABEL[pol]}")
    print(f"{'='*60}")
    print(f"Shapiro-Wilk (Normalitas):")
    for s in sw_results:
        print(f"  {s}")
    print(f"Levene (Homogenitas Varians): stat={stat_lev:.4f}, p={p_lev:.4f} {'Homogen' if p_lev > 0.05 else 'Tidak Homogen'}")
    print(f"\nOne-way ANOVA: F={f_stat:.4f}, p={p_anova:.4f}")
    
    if p_anova < 0.05:
        print(f"  \u2714 Signifikan (p < 0.05) — Tolak H\u2080")
        print(f"\n  Post-hoc Tukey HSD:")
        print(f"  {tukey}")
    else:
        print(f"  \u2716 Tidak signifikan (p >= 0.05) — Gagal tolak H\u2080")

print(f"\n\n{'='*60}")
print("RINGKASAN ANOVA")
print(f"{'='*60}")
print(f"{'Polutan':<8} {'F-stat':<10} {'p-value':<10} {'Levene p':<10} {'Kesimpulan':<25}")
print(f"{'-'*60}")
for r in results:
    kesimpulan = 'Signifikan' if r['p_anova'] < 0.05 else 'Tidak Signifikan'
    label = POLUTAN_LABEL[r['polutan']]
    print(f"{label:<8} {r['f_stat']:<10.4f} {r['p_anova']:<10.4f} {r['levene_p']:<10.4f} {kesimpulan:<25}")
