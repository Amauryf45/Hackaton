from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split


# 1. Charger les données
data = pd.read_csv('data_with_corrected_labels.csv')
data['Activity_Label'] = data['Activity_Label'].fillna('None')

# 2. Prétraitement
# Encodage des étiquettes
le = LabelEncoder()
data['Activity_Label'] = le.fit_transform(data['Activity_Label'])

# Caractéristiques (X) et étiquettes (y)
X = data[['Acc_X_R', 'Acc_Y_R', 'Acc_Z_R', 'Acc_X_L', 'Acc_Y_L', 'Acc_Z_L']]
y = data['Activity_Label']

# Standardisation des caractéristiques
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialisation du modèle
lgb_model = LGBMClassifier(random_state=42, class_weight="balanced")

# Entraînement du modèle
lgb_model.fit(X_train, y_train)

# Prédictions
y_pred_lgb = lgb_model.predict(X_test)

# Évaluation
print("=== LightGBM ===")
print("Classification Report:\n", classification_report(y_test, y_pred_lgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgb))


# Assuming 'solution' is the true labels for the test set
solution = pd.DataFrame({'activity': y_test})
prediction_file = pd.DataFrame({'activity': y_pred_lgb})

print("F-beta Score:", fbeta_score(solution.activity, prediction_file.activity, 
            average='micro', beta=1/3))



none_class = le.transform(['None'])[0]
# 4. Filtrer les classes autres que 'None'
mask = y_test != none_class
filtered_y_test = y_test[mask]
filtered_y_pred = y_pred_lgb[mask]

# Évaluation sur les classes restantes
print("=== LightGBM (Excluant 'None') ===")
print("Classification Report:\n", classification_report(filtered_y_test, filtered_y_pred))
print("Confusion Matrix:\n", confusion_matrix(filtered_y_test, filtered_y_pred))

# Calcul du F-beta Score
f_beta = fbeta_score(filtered_y_test, filtered_y_pred, average='micro', beta=1/3)
print("F-beta Score (Excluant 'None'):", f_beta)