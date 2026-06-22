import json

with open("archive/ml_model/notebooks/compare_classification.ipynb", "r") as f:
    nb = json.load(f)

# =============================================================
# REWRITE CELL [3] — use klhk_combined.csv instead of individual files
# =============================================================
new_cell3 = """SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    env_path = ".env"
    if os.path.exists(env_path):
        for line in open(env_path):
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip(chr(34))
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

SUPABASE_URL = SUPABASE_URL.replace("http://", "https://")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── KLHK ──
# Load pre-combined KLHK data (sudah includes pm25 ISPU index)
KLHK_PATH = "../../../data_train/klhk_combined.csv"
raw = pd.read_csv(KLHK_PATH)
print(f"KLHK raw: {len(raw)} baris")

LABEL_MAP = {
    "BAIK": "Baik", "SEDANG": "Sedang", "TIDAK SEHAT": "Tidak Sehat",
    "SANGAT TIDAK SEHAT": "Sangat Tidak Sehat", "BERBAHAYA": "Berbahaya",
}

def ispi_to_conc(ispi, bp):
    if ispi <= 0: return 0
    for cl, ch, il, ih in bp:
        if ispi <= ih: return cl + (ispi - il) / (ih - il) * (ch - cl)
    return bp[-1][1]

# Breakpoints per PermenLHK 14/2020 (5 range, PM2.5 first C_high=15.5, semua dalam ug/m3)
BP_PM25 = [(0,15.5,0,50),(15.5,55.4,50,100),(55.4,150.4,100,200),(150.4,250.4,200,300),(250.4,500,300,500)]
BP_PM10 = [(0,50,0,50),(50,150,50,100),(150,350,100,200),(350,420,200,300),(420,500,300,500)]
BP_CO   = [(0,4000,0,50),(4000,8000,50,100),(8000,15000,100,200),(15000,30000,200,300),(30000,45000,300,500)]
BP_NO2  = [(0,80,0,50),(80,200,50,100),(200,1130,100,200),(1130,2260,200,300),(2260,3000,300,500)]
BP_O3   = [(0,120,0,50),(120,235,50,100),(235,400,100,200),(400,800,200,300),(800,1000,300,500)]

klhk_rows = []
for _, r in raw.iterrows():
    cat_col = "kategori" if "kategori" in raw.columns and "categori" not in raw.columns else "categori"
    if cat_col not in r or pd.isna(r.get(cat_col)): continue
    label = LABEL_MAP.get(str(r[cat_col]).strip().upper())
    if not label: continue
    # ISPU index columns
    pm25_ispi = pd.to_numeric(r.get("pm25", np.nan), errors="coerce")
    pm10_ispi = pd.to_numeric(r.get("pm10", r.get("pm_10", np.nan)), errors="coerce")
    co_ispi   = pd.to_numeric(r.get("co", np.nan), errors="coerce")
    o3_ispi   = pd.to_numeric(r.get("o3", np.nan), errors="coerce")
    no2_ispi  = pd.to_numeric(r.get("no2", np.nan), errors="coerce")
    if pd.isna(pm25_ispi) or pd.isna(pm10_ispi) or pd.isna(o3_ispi):
        continue
    # Convert ISPU index -> ug/m3
    pm25 = ispi_to_conc(pm25_ispi, BP_PM25)
    pm10 = ispi_to_conc(pm10_ispi, BP_PM10)
    co   = ispi_to_conc(co_ispi if pd.notna(co_ispi) else 0, BP_CO)
    no2  = ispi_to_conc(no2_ispi if pd.notna(no2_ispi) else 0, BP_NO2)
    o3   = ispi_to_conc(o3_ispi, BP_O3)
    klhk_rows.append({"PM2.5": pm25, "PM10": pm10, "CO": co, "NO2": no2, "O3": o3, "Label": label, "src": "klhk"})

df_klhk = pd.DataFrame(klhk_rows)
print(f"KLHK clean: {len(df_klhk)} baris")
print(f"  kategori: {df_klhk['Label'].value_counts().to_dict()}")
"""

nb["cells"][3]["source"] = new_cell3.splitlines(True)

with open("archive/ml_model/notebooks/compare_classification.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Fixed Cell [3]: using klhk_combined.csv")
