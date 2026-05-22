import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = '../data/Clinical-Metadata-FDG-PET_CT-Lesions.csv'

# 1. Caricamento del dataset
df = pd.read_csv(csv_path)

# 2. Preparazione e pulizia dei dati
df_subset = df[['Subject ID', 'diagnosis', 'age']].copy()
df_subset.dropna(subset=['Subject ID', 'diagnosis', 'age'], inplace=True)

# Manteniamo un solo record per paziente
df_unique = df_subset.drop_duplicates(subset=['Subject ID']).copy()

# Filtriamo per Linfoma e Negativi
df_filtered = df_unique[df_unique['diagnosis'].isin(['NEGATIVE', 'LYMPHOMA'])].copy()

# Estraiamo l'età numerica ignorando le lettere (es. '066Y' -> 66)
df_filtered['age'] = df_filtered['age'].str.extract(r'(\d+)').astype(float)
df_filtered.dropna(subset=['age'], inplace=True)

# 3. Creazione della figura con due grafici (1 riga, 2 colonne)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Grafico 1: Linfoma (a sinistra)
sns.histplot(
    data=df_filtered[df_filtered['diagnosis'] == 'LYMPHOMA'], 
    x='age', 
    color='salmon', 
    ax=axes[0],
    bins=15
)
axes[0].set_title('Distribuzione Età - Linfoma')
axes[0].set_xlabel('Età (Anni)')
axes[0].set_ylabel('Numero di Pazienti')
axes[0].grid(axis='y', alpha=0.7)

# Grafico 2: Negativi (a destra)
sns.histplot(
    data=df_filtered[df_filtered['diagnosis'] == 'NEGATIVE'], 
    x='age', 
    color='skyblue', 
    ax=axes[1],
    bins=15
)
axes[1].set_title('Distribuzione Età - Negativi')
axes[1].set_xlabel('Età (Anni)')
axes[1].set_ylabel('') # Nascondiamo la label Y qui dato che è condivisa a sinistra
axes[1].grid(axis='y', alpha=0.7)

# Aggiungiamo un titolo principale alla figura intera
plt.suptitle("Confronto Distribuzione dell'Età: Linfoma vs Negativi", fontsize=16)

# Regoliamo gli spazi in modo che niente si sovrapponga e salviamo
plt.tight_layout()
plt.savefig('distribuzione_eta_separata.png', bbox_inches='tight')
plt.show()