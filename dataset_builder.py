import os
import shutil
import random
from pathlib import Path

# ==========================================
# CONFIGURAZIONE
# ==========================================

CARTELLA_SORGENTE = Path("./data/Lymphoma")
CARTELLA_DESTINAZIONE = Path("./data/Lymphoma_segmentation")

NOME_FILE_PET = "SUV.nii.gz"
NOME_FILE_CT = "CTres.nii.gz"
NOME_FILE_MASK = "SEG.nii.gz"

# Percentuali split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed per split riproducibile
RANDOM_SEED = 42

# ==========================================
# CREAZIONE CARTELLE
# ==========================================

imagesTr_path = CARTELLA_DESTINAZIONE / "imagesTr"
labelsTr_path = CARTELLA_DESTINAZIONE / "labelsTr"

imagesVal_path = CARTELLA_DESTINAZIONE / "imagesVal"
labelsVal_path = CARTELLA_DESTINAZIONE / "labelsVal"

imagesTs_path = CARTELLA_DESTINAZIONE / "imagesTs"
labelsTs_path = CARTELLA_DESTINAZIONE / "labelsTs"

for p in [
    imagesTr_path,
    labelsTr_path,
    imagesVal_path,
    labelsVal_path,
    imagesTs_path,
    labelsTs_path,
]:
    p.mkdir(parents=True, exist_ok=True)

# ==========================================
# RACCOLTA PAZIENTI VALIDI
# ==========================================

maschere_trovate = sorted(CARTELLA_SORGENTE.rglob(NOME_FILE_MASK))

pazienti_validi = []

for mask_path in maschere_trovate:

    parent_folder = mask_path.parent.name.replace(" ", "_")
    grandparent_folder = mask_path.parent.parent.name.replace(" ", "_")

    patient_id = f"{grandparent_folder}_{parent_folder}"

    pet_path = mask_path.parent / NOME_FILE_PET
    ct_path = mask_path.parent / NOME_FILE_CT

    if not pet_path.exists() or not ct_path.exists():
        print(f"⚠️ Salto {patient_id}: PET o CT mancante.")
        continue

    pazienti_validi.append({
        "id": patient_id,
        "pet": pet_path,
        "ct": ct_path,
        "mask": mask_path
    })

print(f"\n✅ Pazienti validi trovati: {len(pazienti_validi)}")

# ==========================================
# SHUFFLE + SPLIT
# ==========================================

random.seed(RANDOM_SEED)
random.shuffle(pazienti_validi)

n_total = len(pazienti_validi)

n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_set = pazienti_validi[:n_train]
val_set = pazienti_validi[n_train:n_train + n_val]
test_set = pazienti_validi[n_train + n_val:]

print("\n📊 SPLIT DATASET")
print(f"Train: {len(train_set)}")
print(f"Validation: {len(val_set)}")
print(f"Test: {len(test_set)}")

# ==========================================
# FUNZIONE COPIA
# ==========================================

def copia_pazienti(lista_pazienti, images_path, labels_path, split_name):

    for paziente in lista_pazienti:

        patient_id = paziente["id"]

        ct_path = paziente["ct"]
        pet_path = paziente["pet"]
        mask_path = paziente["mask"]

        # CT -> _0000
        nuovo_nome_ct = images_path / f"{patient_id}_0000.nii.gz"
        shutil.copy2(ct_path, nuovo_nome_ct)

        # PET -> _0001
        nuovo_nome_pet = images_path / f"{patient_id}_0001.nii.gz"
        shutil.copy2(pet_path, nuovo_nome_pet)

        # MASK
        nuovo_nome_mask = labels_path / f"{patient_id}.nii.gz"
        shutil.copy2(mask_path, nuovo_nome_mask)

        print(f"✅ [{split_name}] {patient_id}")

# ==========================================
# COPIA FILE
# ==========================================

print("\n🚀 Copia TRAIN")
copia_pazienti(train_set, imagesTr_path, labelsTr_path, "TRAIN")

print("\n🚀 Copia VALIDATION")
copia_pazienti(val_set, imagesVal_path, labelsVal_path, "VAL")

print("\n🚀 Copia TEST")
copia_pazienti(test_set, imagesTs_path, labelsTs_path, "TEST")

# ==========================================
# FINE
# ==========================================

print("\n" + "="*50)
print("🎉 FORMATTAZIONE COMPLETATA!")
print("="*50)

print(f"Train samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")
print(f"Test samples: {len(test_set)}")