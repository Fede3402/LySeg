import os 
import pandas as pd 

def prepare_patients_list(csv_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)

    # Extraction of age info with numerical format (ex. 065Y)
    df['age_num'] = df['age'].str.extract(r'(\d+)').astype(float)

    # Multiple patients scan handled sorting all scans in chronological order
    df['Study Date'] = pd.to_datetime(df['Study Date'], errors='coerce')
    df_sorted = df.sort_values(by=['Subject ID', 'Study Date'])

    # Only the first line is kept
    df_unique = df_sorted.drop_duplicates(subset=['Subject ID'], keep='first')

    # Filter based on negative patient and not in pediatric age
    df_neg = df_unique[df_unique['diagnosis'] == 'NEGATIVE'].copy()
    df_adults = df_neg[df_neg['age_num'] >= 18].copy()

    # Group creation
    group1 = df_adults[df_adults['age_num'] < 41]
    group2 = df_adults[(df_adults['age_num'] >= 41)]

    print(f'Total number of valid patients: {len(df_adults)}')
    print(f'Number of patients in group 1 (18-40): {len(group1)}')
    print(f'Number of patients in group 2 (40+): {len(group2)}')

    g1_ids = group1['Subject ID'].tolist()
    g2_ids = group2['Subject ID'].tolist()

    with open(os.path.join(output_dir, "atlas_18_40_ids.txt"), 'w') as f:
        f.write('\n'.join(g1_ids))
    with open(os.path.join(output_dir, "atlas_41_plus_ids.txt"), 'w') as f:
        f.write('\n'.join(g2_ids))

    print('Lists created successfully!!')

if __name__ == "__main__": 

    csv_path = '../../data/Clinical-Metadata-FDG-PET_CT-Lesions.csv'
    output_dir = '../../data/Metadata for atlas creation'
    
    prepare_patients_list(csv_path, output_dir)




