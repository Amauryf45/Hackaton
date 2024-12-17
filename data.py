import pandas as pd

# Charger les fichiers CSV pour la main droite et la main gauche
dataRight = pd.read_csv('right_accs.csv', header=None, names=['Time', 'Acc_X_R', 'Acc_Y_R', 'Acc_Z_R'])
dataLeft = pd.read_csv('left_accs.csv', header=None, names=['Time', 'Acc_X_L', 'Acc_Y_L', 'Acc_Z_L'])

# Convertir la colonne 'Time' en datetime
dataRight['Time'] = pd.to_datetime(dataRight['Time'], unit='s')
dataLeft['Time'] = pd.to_datetime(dataLeft['Time'], unit='s')

# Fusionner les données sur 'Time'
data = pd.merge(dataRight, dataLeft, on='Time', how='inner')

# Arrondir les temps à la seconde
data['Time_Second'] = data['Time'].dt.floor('S')

# Calculer la moyenne pour chaque seconde
data_avg_byS = data.groupby('Time_Second').mean().reset_index()

# Créer un index temporel complet basé sur la plage des données existantes
start_time = data['Time_Second'].min()
end_time = data['Time_Second'].max()
time_index = pd.date_range(start=start_time, end=end_time, freq='S')

# Réindexer pour inclure toutes les secondes dans la plage
data_avg_byS = data_avg_byS.set_index('Time_Second').reindex(time_index).reset_index()
data_avg_byS.rename(columns={'index': 'Time_Second'}, inplace=True)

# Remplir les valeurs manquantes avec les moyennes précédentes ou des NaN
data_avg_byS.fillna(method='ffill', inplace=True)  # Vous pouvez utiliser 'bfill' ou une valeur par défaut si nécessaire

# Exporter les données moyennées par seconde
data_avg_byS.to_csv('data_average_by_second.csv', index=False)

# Afficher un aperçu des données
print(data_avg_byS.head())
