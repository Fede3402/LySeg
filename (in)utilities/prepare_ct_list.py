# check_patients_on_disk.py
import pandas as pd
from pathlib import Path

# ── CONFIGURAZIONE ───────────────────────────────────────
METADATA_CSV = "../data/Clinical-Metadata-FDG-PET_CT-Lesions.csv"
DATA_DIR     = "../data/Negative_Light"
OUTPUT_CSV   = "./results/patients_on_disk.csv"
# ────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(METADATA_CSV)

    # Filtra solo CT e negativi
    ct_df = df[df['Modality'] == 'CT'].copy()
    ct_df['age_int'] = ct_df['age'].str.extract(r'(\d+)').astype(int)

    # Deduplica per paziente
    ct_df = ct_df.sort_values('Study Date')
    ct_unique = ct_df.drop_duplicates(subset='Subject ID', keep='first')

    negatives = ct_unique[ct_unique['diagnosis'] == 'NEGATIVE'].copy()
    print(f"Negativi totali nel CSV: {len(negatives)}")

    # Controlla presenza su disco
    results = []
    for _, row in negatives.iterrows():
        subject_id = row['Subject ID']
        patient_dir = Path(DATA_DIR) / subject_id

        # Cerca CTres.nii.gz
        matches = sorted(patient_dir.rglob("SUV.nii.gz")) \
                  if patient_dir.exists() else []
        on_disk = len(matches) > 0

        results.append({
            'subject_id': subject_id,
            'age':        row['age_int'],
            'sex':        row['sex'],
            'on_disk':    on_disk,
            'ct_path':    str(matches[0]) if on_disk else '',
        })

    # Report
    result_df = pd.DataFrame(results)
    on_disk   = result_df[result_df['on_disk']]
    missing   = result_df[~result_df['on_disk']]

    print(f"\nPresenti su disco: {len(on_disk)}")
    print(f"Mancanti:          {len(missing)}")

    # Distribuzione età di quelli sul disco
    print(f"\n=== Anagrafica pazienti su disco ===")
    print(f"Età media:     {on_disk['age'].mean():.1f}")
    print(f"Giovani (≤40): {(on_disk['age'] <= 40).sum()}")
    print(f"Adulti  (>40): {(on_disk['age'] > 40).sum()}")
    print(f"Sesso M:       {(on_disk['sex'] == 'M').sum()}")
    print(f"Sesso F:       {(on_disk['sex'] == 'F').sum()}")

    # Salva
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRisultato completo salvato: {OUTPUT_CSV}")

    # Salva solo quelli su disco
    on_disk.to_csv("./results/patients_on_disk_only.csv", index=False)
    print(f"Solo quelli su disco: patients_on_disk_only.csv")

if __name__ == "__main__":
    main()