import json

with open(
    r"D:\DashboardAQ_v3\archive\ml_model\notebooks\analisis_spasial.ipynb",
    "r",
    encoding="utf-8",
) as f:
    nb = json.load(f)

nb["cells"][10]["source"] = [
    "# Label segmen\n",
    "segmen_labels = []\n",
    "seg = 0\n",
    "for i in range(len(df)):\n",
    "    if i in gap_indices:\n",
    "        seg += 1\n",
    "    segmen_labels.append(seg)\n",
    "\n",
    "df['segmen'] = segmen_labels\n",
    "\n",
    "# Segmen 0 hanya 1 data, digabung ke segmen 1\n",
    "df['segmen'] = df['segmen'].replace(0, 1)\n",
    "\n",
    "# Trim setiap segmen ke 60 data terbaru\n",
    "segments_trimmed = []\n",
    "for s in sorted(df['segmen'].unique()):\n",
    "    sub = df[df['segmen'] == s].tail(60)\n",
    "    segments_trimmed.append(sub)\n",
    "df = pd.concat(segments_trimmed).sort_values('created_at').reset_index(drop=True)\n",
    "\n",
    "lokasi_map = {\n",
    "    1: 'Ngagel (Pagi - Urban Traffic)',\n",
    "    2: 'Rungkut (Siang - Industri Manufaktur)',\n",
    "    3: 'Margomulyo (Sore - Industri Logistik)'\n",
    "}\n",
    "df['lokasi'] = df['segmen'].map(lokasi_map)\n",
    "\n",
    "for s in sorted(df['segmen'].unique()):\n",
    "    sub = df[df['segmen'] == s]\n",
    "    print(f'{lokasi_map[s]}:')\n",
    '    print(f\'  Jumlah: {len(sub)}, {sub["created_at_wib"].iloc[0]} - {sub["created_at_wib"].iloc[-1]}\')\n',
    "    print()\n",
]

with open(
    r"D:\DashboardAQ_v3\archive\ml_model\notebooks\analisis_spasial.ipynb",
    "w",
    encoding="utf-8",
) as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cell 10 updated")
