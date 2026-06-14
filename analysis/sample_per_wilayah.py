import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
IN_CSV = BASE / "hasil_klasifikasi_spasial_rf.csv"
OUT_CSV = BASE / "sampled_60_per_wilayah.csv"


def segment_by_time(df, gap_seconds=3600):
    df = df.sort_values("created_at").reset_index(drop=True)
    diffs = df["created_at"].diff().dt.total_seconds()
    gap_idx = diffs[diffs > gap_seconds].index.tolist()
    seg = 1
    segs = []
    for i in range(len(df)):
        if i in gap_idx:
            seg += 1
        segs.append(seg)
    df["segmen"] = segs
    return df


def map_lokasi(df):
    # Map segments to lokasi using fixed time windows observed in analysis notebook.
    # These windows were identified in the notebook as:
    # 1: Ngagel -> 2026-06-09 23:39:21 .. 2026-06-10 02:06:03
    # 2: Rungkut -> 2026-06-10 05:25:06 .. 2026-06-10 06:47:30
    # 3: Margomulyo -> 2026-06-10 08:47:27 .. 2026-06-10 09:49:39
    import pandas as pd

    windows = [
        (
            "Ngagel",
            pd.to_datetime("2026-06-09 23:39:21+00:00"),
            pd.to_datetime("2026-06-10 02:06:03+00:00"),
        ),
        (
            "Rungkut",
            pd.to_datetime("2026-06-10 05:25:06+00:00"),
            pd.to_datetime("2026-06-10 06:47:30+00:00"),
        ),
        (
            "Margomulyo",
            pd.to_datetime("2026-06-10 08:47:27+00:00"),
            pd.to_datetime("2026-06-10 09:49:39+00:00"),
        ),
    ]

    def assign_lokasi(ts):
        for name, start, end in windows:
            if ts >= start and ts <= end:
                return name
        return "unknown"

    df["lokasi"] = df["created_at"].apply(assign_lokasi)
    return df


def stratified_sample_per_lokasi(df, n_per_lokasi=60, random_state=42):
    out_rows = []
    for lokasi in sorted(df["lokasi"].unique()):
        sub = df[df["lokasi"] == lokasi].copy()
        if len(sub) == 0:
            continue

        # For Ngagel, ensure all 'Tidak Sehat' are included
        if lokasi == "Ngagel":
            tidak_sehat = sub[sub["predicted_category"] == "Tidak Sehat"]
            remaining = sub.drop(tidak_sehat.index)
            need = n_per_lokasi - len(tidak_sehat)
            if need <= 0:
                sampled = tidak_sehat.sample(n=n_per_lokasi, random_state=random_state)
            else:
                # If remaining smaller than need, sample with replacement to fill up
                if len(remaining) >= need:
                    sampled_rest = remaining.sample(n=need, random_state=random_state)
                else:
                    sampled_rest = remaining.sample(
                        n=need, replace=True, random_state=random_state
                    )
                sampled = pd.concat([tidak_sehat, sampled_rest]).sample(
                    frac=1, random_state=random_state
                )
        else:
            # simple random sample n_per_lokasi (if available)
            if len(sub) >= n_per_lokasi:
                sampled = sub.sample(n=n_per_lokasi, random_state=random_state)
            else:
                # if less than needed, take all
                sampled = sub.copy()

        out_rows.append(sampled)

    result = pd.concat(out_rows).reset_index(drop=True)
    return result


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["created_at"])
    df = segment_by_time(df)
    df = map_lokasi(df)

    # Exclude rows that could not be assigned to a lokasi (unknown)
    unknown_count = (df["lokasi"] == "unknown").sum()
    if unknown_count > 0:
        print(f"Dropping {unknown_count} rows with lokasi='unknown'")
    df = df[df["lokasi"] != "unknown"]

    sampled = stratified_sample_per_lokasi(df, n_per_lokasi=60)

    # Summary
    print("Samples per lokasi:")
    print(sampled["lokasi"].value_counts())
    print("\nClass distribution in Ngagel (predicted_category):")
    print(sampled[sampled["lokasi"] == "Ngagel"]["predicted_category"].value_counts())

    sampled.to_csv(OUT_CSV, index=False)
    print(f"Wrote sampled dataset to: {OUT_CSV}")


if __name__ == "__main__":
    main()
