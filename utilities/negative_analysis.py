import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# --- CONFIGURAZIONE ---
# Inserisci il percorso della cartella dove hai ESTRATTO lo ZIP dei 50 negativi
CARTELLA_ESTRATTA = Path("../Risultati_LION_Negativi")

print(f"🔍 Scansione in corso su: {CARTELLA_ESTRATTA.name}...")

# Trova tutte le maschere ovunque siano nelle sottocartelle
maschere = list(CARTELLA_ESTRATTA.rglob('lesion_mask_none.nii.gz'))

if not maschere:
    print("❌ Nessuna maschera trovata! Controlla di aver estratto lo ZIP e di aver messo il percorso giusto.")
else:
    print(f"📁 Trovate {len(maschere)} maschere. Inizio analisi...\n")
    
    veri_negativi = 0
    falsi_positivi = []
    
    # Analizziamo ogni paziente
    for m in maschere:
        scan_id = m.parent.name
        
        # Leggiamo l'immagine
        img = sitk.ReadImage(str(m))
        array = sitk.GetArrayFromImage(img)
        
        # Contiamo i voxel che sono stati classificati come tumore (>0)
        voxel_attivi = np.sum(array > 0)
        
        if voxel_attivi == 0:
            veri_negativi += 1
        else:
            # Calcoliamo anche il volume in mL per capire "quanto" ha sbagliato
            spacing = img.GetSpacing()
            volume_ml = (voxel_attivi * spacing[0] * spacing[1] * spacing[2]) / 1000.0
            falsi_positivi.append((scan_id, voxel_attivi, volume_ml))
    
    # --- REPORT FINALE ---
    print("="*60)
    print("📊 REPORT SPECIFICITÀ (PAZIENTI NEGATIVI)")
    print("="*60)
    print(f"Totale pazienti analizzati: {len(maschere)}")
    print(f"✅ Veri Negativi (Maschere 100% vuote): {veri_negativi}")
    print(f"⚠️ Falsi Positivi (Allucinazioni di LION): {len(falsi_positivi)}\n")
    
    specificita = (veri_negativi / len(maschere)) * 100
    print(f"🎯 Specificità Attuale (Senza Filtri): {specificita:.2f}%")
    print("="*60)
    
    # Dettaglio degli errori per capire se si possono filtrare
    if falsi_positivi:
        print("\n📋 DETTAGLIO FALSI POSITIVI (Dove LION ha sbagliato):")
        # Ordiniamo i falsi positivi dal più piccolo al più grande
        falsi_positivi.sort(key=lambda x: x[1]) 
        
        for fp in falsi_positivi:
            scan_id, voxel, ml = fp
            print(f"  - {scan_id}: {voxel} voxel ({ml:.2f} mL)")
        
        print("\n💡 Suggerimento Clinico:")
        print("Se i Falsi Positivi hanno volumi molto piccoli (es. meno di 1-2 mL),")
        print("significa che LION ha preso solo del 'rumore' di fondo.")
        print("Applicando un Filtro di Volume Minimo, la tua Specificità schizzerà al 100%.")
