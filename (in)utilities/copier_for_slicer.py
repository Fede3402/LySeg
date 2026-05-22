import shutil
from pathlib import Path

# 1. Definisci la cartella di partenza (la radice del tuo dataset originale)
cartella_radice = Path("./data/Lymphoma")

# 2. Definisci la cartella dove vuoi COPIARE tutti i file
cartella_destinazione = Path("ct_for_slicer")
cartella_destinazione.mkdir(parents=True, exist_ok=True)

# 3. Cerca ricorsivamente tutti i file 'CTres.nii.gz'
conteggio = 0

for file_origine in cartella_radice.rglob("CTres.nii.gz"):
    # Recuperiamo i due livelli superiori del percorso:
    cartella_esame = file_origine.parts[-2]   # es. '11-16-2002-NA-PET-CT...'
    id_paziente = file_origine.parts[-3]      # es. 'PETCT_9f6e8b1b43'
    
    # Creiamo un nome unico che include sia il paziente che l'esame per evitare sovrascritture
    # Risultato: "PETCT_9f6e8b1b43__11-16-2002-NA-PET-CT...__CTres.nii.gz"
    nuovo_nome = f"{id_paziente}__{cartella_esame}__{file_origine.name}"
    file_destinazione = cartella_destinazione / nuovo_nome
    
    # Copiamo il file mantenendo i metadati originali
    shutil.copy2(str(file_origine), str(file_destinazione))
    
    print(f"Copiato [{conteggio + 1}]: {id_paziente} -> {cartella_esame}")
    conteggio += 1

print(f"\nOperazione completata! Copiati correttamente {conteggio} file nella cartella '{cartella_destinazione.name}'.")