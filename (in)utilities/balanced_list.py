# generate_balanced_list.py
# generate_balanced_list.py
import pandas as pd
import shutil
from pathlib import Path

PATIENTS_CSV  = "./results/patients_on_disk_only.csv"
OUTPUT_TXT    = "./results/ct_list_balanced.txt"
#OUTPUT_YOUNG  = "slicer_young"   # cartella giovani
OUTPUT_ADULTS = "slicer_adults"  # cartella adulti
RANDOM_SEED   = 42

def copy_patient(row, dest_dir):
    """Copia CT e SUV nella cartella destinazione con patient_id nel nome."""
    subject_id = row['subject_id']
    ct_path    = Path(row['ct_path'])
    pet_path   = ct_path.parent / "SUV.nii.gz"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copia CT
    ct_dest = dest_dir / f"CT_{subject_id}.nii.gz"
    shutil.copy2(ct_path, ct_dest)

    # Copia SUV se esiste
    if pet_path.exists():
        suv_dest = dest_dir / f"SUV_{subject_id}.nii.gz"
        shutil.copy2(pet_path, suv_dest)
    else:
        print(f"  ATTENZIONE: SUV non trovato per {subject_id}")

def main():
    df = pd.read_csv(PATIENTS_CSV)

    # Dividi per età
    # young  = df[df['age'] <= 40]
    adults = df[df['age'] > 40]

    #print(f"Giovani (≤40): {len(young)}")
    print(f"Adulti  (>40): {len(adults)}")

    # Prendi tutti i giovani e lo stesso numero di adulti
    #n = len(young)  # 26
    adults_sample = adults.sample(n=50, random_state=RANDOM_SEED)

    #selected_young  = young
    selected_adults = adults_sample
    
    #print(f"Selezionati: {n} giovani + {n} adulti = {n*2} totali")

    # Cartelle output per Slicer
    #young_dir  = Path(OUTPUT_YOUNG)
    adults_dir = Path(OUTPUT_ADULTS)
    '''
    # Processa giovani
    print(f"\n=== Copia giovani → {young_dir} ===")
    valid_paths = []
    missing     = []

    for _, row in selected_young.iterrows():
        ct_path = Path(row['ct_path'])
        if not ct_path.exists():
            missing.append(row['subject_id'])
            print(f"  NON TROVATO: {row['subject_id']}")
            continue
        print(f"  {row['subject_id']} (età {row['age']}, {row['sex']})")
        copy_patient(row, young_dir)
        valid_paths.append(str(ct_path))
    '''

    valid_paths = []
    missing     = []

    # Processa adulti
    print(f"\n=== Copia adulti → {adults_dir} ===")
    for _, row in selected_adults.iterrows():
        ct_path = Path(row['ct_path'])
        if not ct_path.exists():
            missing.append(row['subject_id'])
            print(f"  NON TROVATO: {row['subject_id']}")
            continue
        print(f"  {row['subject_id']} (età {row['age']}, {row['sex']})")
        copy_patient(row, adults_dir)
        valid_paths.append(str(ct_path))

    # Report
    print(f"\n=== Report finale ===")
    print(f"Copiati con successo: {len(valid_paths)}/{50*2}")
    if missing:
        print(f"Mancanti: {missing}")

    #print(f"\nContenuto {young_dir}:")
    #print(f"  CT:  {len(list(young_dir.glob('CT_*.nii.gz')))} file")
    #print(f"  SUV: {len(list(young_dir.glob('SUV_*.nii.gz')))} file")

    print(f"Contenuto {adults_dir}:")
    print(f"  CT:  {len(list(adults_dir.glob('CT_*.nii.gz')))} file")
    print(f"  SUV: {len(list(adults_dir.glob('SUV_*.nii.gz')))} file")

    # Salva txt per ANTs (solo CT, percorsi originali)
    with open(OUTPUT_TXT, 'w') as f:
        for path in valid_paths:
            f.write(path + '\n')

    print(f"\nLista per ANTs salvata: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()