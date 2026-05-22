# find_missing_young.py
import pandas as pd
from pathlib import Path

METADATA_CSV  = "./results/patients_on_disk.csv"  # output del primo script

def main():
    df = pd.read_csv(METADATA_CSV)
    
    # Giovani non sul disco
    missing_young = df[
        (df['age'] <= 40) & 
        (df['on_disk'] == False)
    ][['subject_id', 'age', 'sex']]
    
    print(f"Giovani (≤40) mancanti sul disco: {len(missing_young)}")
    print()
    for _, row in missing_young.iterrows():
        print(f"  {row['subject_id']}  età: {row['age']}  sesso: {row['sex']}")

if __name__ == "__main__":
    main()