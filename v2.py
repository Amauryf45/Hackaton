from datetime import datetime
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier  # Ajout de LightGBM pour l'enssemble

# 1. Charger les données
data = pd.read_csv('data_with_corrected_labels.csv')
data['Activity_Label'] = data['Activity_Label'].fillna('None')

# 2. Prétraitement
# Encodage des étiquettes
le = LabelEncoder()
data['Activity_Label'] = le.fit_transform(data['Activity_Label'])

# Caractéristiques (X) et étiquettes (y)
data['Norm_R'] = (data['Acc_X_R']**2 + data['Acc_Y_R']**2 + data['Acc_Z_R']**2)**0.5
data['Norm_L'] = (data['Acc_X_L']**2 + data['Acc_Y_L']**2 + data['Acc_Z_L']**2)**0.5

# Diviser les données en fonction des jours
# On suppose que les lignes sont triées chronologiquement dans le fichier
day_5_data = data.tail(86400)  # Les 86400 dernières lignes (5ème jour)
day_1_to_4_data = data.iloc[:-86400]  # Toutes les lignes sauf les 86400 dernières (jours 1 à 4)

# Séparation des caractéristiques et étiquettes
X_train = day_1_to_4_data[['Acc_X_R', 'Acc_Y_R', 'Acc_Z_R', 'Acc_X_L', 'Acc_Y_L', 'Acc_Z_L', 'Norm_R', 'Norm_L']]
y_train = day_1_to_4_data['Activity_Label']

X_test = day_5_data[['Acc_X_R', 'Acc_Y_R', 'Acc_Z_R', 'Acc_X_L', 'Acc_Y_L', 'Acc_Z_L', 'Norm_R', 'Norm_L']]
y_test = day_5_data['Activity_Label']

# Standardisation des caractéristiques (en utilisant uniquement les données d'entraînement pour fit)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Rééquilibrage des classes avec SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 5. Random Forest avec GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 50],
    'class_weight': ['balanced', 'balanced_subsample']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='precision_weighted', cv=5)
grid_search.fit(X_resampled, y_resampled)

# Évaluer le modèle Random Forest optimisé
best_rf_model = grid_search.best_estimator_
rf_pred = best_rf_model.predict(X_test)
print("=== Random Forest ===")
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# 6. XGBoost
xgb_model = XGBClassifier(learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_resampled, y_resampled)

# Évaluer le modèle XGBoost
xgb_pred = xgb_model.predict(X_test)
print("=== XGBoost ===")
print("Classification Report:\n", classification_report(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

# 7. LightGBM
lgbm_model = LGBMClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')
lgbm_model.fit(X_resampled, y_resampled)

# Évaluer le modèle LightGBM
lgbm_pred = lgbm_model.predict(X_test)
print("=== LightGBM ===")
print("Classification Report:\n", classification_report(y_test, lgbm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lgbm_pred))

# 8. Validation croisée stratifiée pour évaluer les modèles
models = {'Random Forest': best_rf_model, 'XGBoost': xgb_model, 'LightGBM': lgbm_model}
for model_name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_resampled, y_resampled, scoring='precision_weighted', cv=skf)
    print(f"Validation croisée pour {model_name} - Précision moyenne : {scores.mean():.4f}")

# 9. Ensemble VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft'
)
ensemble_model.fit(X_resampled, y_resampled)

# Évaluer le modèle ensemble
ensemble_pred = ensemble_model.predict(X_test)
print("=== Ensemble Voting Classifier ===")
print("Classification Report:\n", classification_report(y_test, ensemble_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ensemble_pred))


none_class = le.transform(['None'])[0]
# 4. Filtrer les classes autres que 'None'
mask = y_test != none_class
filtered_y_test = y_test[mask]
filtered_y_pred = ensemble_pred[mask]

# Évaluation sur les classes restantes
print("=== XGBoost (Excluant 'None') ===")
print("Classification Report:\n", classification_report(filtered_y_test, filtered_y_pred))
print("Confusion Matrix:\n", confusion_matrix(filtered_y_test, filtered_y_pred))

# Calcul du F-beta Score
f_beta = fbeta_score(filtered_y_test, filtered_y_pred, average='micro', beta=1/3)
print("F-beta Score (Excluant 'None'):", f_beta)




data = pd.read_excel("prediction_label_day.csv")

# Vérifier le contenu du fichier
print(data.head())

# Ajouter les colonnes de normes pour les deux mains
data['Norm_R'] = (data['Acc_X_R']**2 + data['Acc_Y_R']**2 + data['Acc_Z_R']**2)**0.5
data['Norm_L'] = (data['Acc_X_L']**2 + data['Acc_Y_L']**2 + data['Acc_Z_L']**2)**0.5


# Créer une plage de temps entre 07:00:00 et 19:00:00 pour chaque seconde
start_time = datetime.strptime("07:00:00", "%H:%M:%S")
end_time = datetime.strptime("19:00:00", "%H:%M:%S")

time_range = pd.date_range(start=start_time, end=end_time, freq="1S")

# Convertir vos données en DataFrame
predictions_df = pd.DataFrame({'time': time_range, 'activity': 'none'}) 


# Prétraiter les données de jour 6 (normalisation des caractéristiques)
# Inclure toutes les colonnes utilisées lors de l'entraînement
features = data[['Acc_X_R', 'Acc_Y_R', 'Acc_Z_R', 'Acc_X_L', 'Acc_Y_L', 'Acc_Z_L', 'Norm_R', 'Norm_L']]

# Standardisation des caractéristiques
scaler = StandardScaler()
scaled_features = scaler.transform(features)

# Faire les prédictions
# Assurez-vous d'avoir chargé votre modèle (par exemple, `ensemble_model` est votre modèle pré-entraîné)
predicted_labels = ensemble_model.predict(scaled_features)

# Ajouter les prédictions au DataFrame
predictions_df['activity'] = predicted_labels

# Si les prédictions sont encodées, les inverser pour obtenir les noms des activités
predictions_df['activity'] = le.inverse_transform(predictions_df['activity'])

# Sauvegarder les prédictions dans un fichier CSV
output_file = "predicted_activities.csv"
predictions_df.to_csv(output_file, index=False)

print(f"Prédictions sauvegardées dans : {output_file}")