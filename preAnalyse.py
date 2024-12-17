import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

# 1. Charger les données
data = pd.read_csv('data_with_corrected_labels.csv')

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

# 4. Modèle de base
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Évaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Assuming 'solution' is the true labels for the test set
solution = pd.DataFrame({'activity': y_test})
prediction_file = pd.DataFrame({'activity': y_pred})

print("F-beta Score:", fbeta_score(solution.activity, prediction_file.activity, 
            average='micro', beta=1/3))