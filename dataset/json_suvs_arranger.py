import os 
import pandas as pd 
import json 
import re

########################### PATH CONFIGURATION ##########################
csv_path = "../data/Lymphoma_segmentation/Advanced_Prior_Stats.csv"
json_output_path = "../data/Lymphoma_segmentation/dataset.json"

directory_splits = {
    
    "training":
    {
        "images":"../data/Lymphoma_segmentation/imagesTr",
        "labels":"../data/Lymphoma_segmentation/labelsTr"
    },

    "validation":
    {
        "images":"../data/Lymphoma_segmentation/imagesVal",
        "labels":"../data/Lymphoma_segmentation/labelsVal"
    },

    "testing":
    {
        "images":"../data/Lymphoma_segmentation/imagesTs",
        "labels":"../data/Lymphoma_segmentation/labelsTs"
    }

}

# Definiamo le 11 colonne da estrarre dal CSV
PRIOR_COLUMNS = [
    "SUV_Mean", "SUV_Median", "SUV_SD", "SUV_P75", "SUV_P90", 
    "SUV_P95", "SUV_Peak_Approx", "Skewness", "Kurtosis", 
    "Entropy", "Liver_Volume_mL"
]
##########################################################################

def clean_id(s):
    """Rimuove underscore, spazi e trattini per un confronto robusto"""
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

def create_json():
    if not os.path.exists(csv_path):
        print(f"ERRORE: CSV non trovato in {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Creiamo una lookup table 'pulita'
    # Chiave: ID pulito, Valore: [ID originale nel CSV, [statistiche]]
    clean_lookup = {}
    for _, row in df.iterrows():
        orig_id = str(row["Scan_ID"]).strip()
        c_id = clean_id(orig_id)
        stats = [float(row[col]) for col in PRIOR_COLUMNS]
        clean_lookup[c_id] = stats

    dataset_dict = {"training": [], "validation": [], "testing": []}
    pazienti_trovati = 0
    pazienti_ignorati = 0

    for split, paths in directory_splits.items():
        img_dir = paths["images"]
        lbl_dir = paths["labels"]

        if not os.path.exists(img_dir):
            continue
            
        for filename in os.listdir(img_dir):
            if filename.endswith("_0000.nii.gz"):
                # 1. Estraiamo l'ID dal nome file e puliamolo
                raw_file_id = filename.replace("_0000.nii.gz", "")
                c_file_id = clean_id(raw_file_id)
                
                # 2. Cerchiamo nel dizionario pulito
                if c_file_id in clean_lookup:
                    prior_list = clean_lookup[c_file_id]
                    
                    pet_path = os.path.join(img_dir, filename)
                    ct_path = os.path.join(img_dir, filename.replace("_0000", "_0001"))
                    # Cerchiamo la label (spesso non ha _0000)
                    label_path = os.path.join(lbl_dir, filename.replace("_0000", ""))
                    
                    dataset_dict[split].append({
                        "scan_id": raw_file_id,
                        "image": [pet_path, ct_path],
                        "label": label_path,
                        "prior": prior_list
                    })
                    pazienti_trovati += 1
                else:
                    print(f"⚠️ Nessun match per: {raw_file_id}")
                    pazienti_ignorati += 1

    with open(json_output_path, 'w') as f:
        json.dump(dataset_dict, f, indent=4) 

    print(f"\n✅ Generazione JSON completata!")
    print(f"   - Totale pazienti con match: {pazienti_trovati}")
    print(f"   - Totale pazienti saltati:   {pazienti_ignorati}")
    for k, v in dataset_dict.items():
        print(f"     -> {k}: {len(v)}")

if __name__ == "__main__":
    create_json()