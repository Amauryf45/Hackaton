import pandas as pd
from datetime import timedelta

# Charger les données temporelles
data_avg_byS = pd.read_csv('data_average_by_second.csv')

# Charger le fichier de labels
labels = pd.read_csv('combined_activity_journal.csv')

# Convertir les colonnes de temps en datetime
data_avg_byS['Time'] = pd.to_datetime(data_avg_byS['Time'])
labels['Start'] = pd.to_datetime(labels['start_time'])
labels['End'] = pd.to_datetime(labels['end_time'])

# Ajouter un décalage pour corriger les dates des labels
date_offset = pd.Timestamp("2024-07-19") - pd.Timestamp("1900-01-01")
labels['Start'] += date_offset
labels['End'] += date_offset

# Ajouter une colonne 'Activity_Label' initialement à 'None'
data_avg_byS['Activity_Label'] = "None"

# Appliquer les labels en fonction des intervalles
for _, row in labels.iterrows():
    start = row['Start']
    end = row['End']
    activity = row['activity']
    
    # Filtrer les lignes de data_avg_byS dans l'intervalle [Start, End]
    data_avg_byS.loc[(data_avg_byS['Time'] >= start) & (data_avg_byS['Time'] <= end), 'Activity_Label'] = activity

# Exporter les données combinées avec les labels
data_avg_byS.to_csv('data_with_corrected_labels_withEmptyDays.csv', index=False)

# Afficher un aperçu des données combinées
print(data_avg_byS.head())

