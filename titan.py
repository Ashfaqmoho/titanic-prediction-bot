import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit as ssf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


#Load data
titan_data = pd.read_csv('train.csv', encoding='latin-1')

#Filling the missing data
titan_data['Age'] = titan_data['Age'].fillna(titan_data['Age'].median())

#Dropping unwanted columns
dp_list = ['Name', 'Embarked', 'Cabin', 'Ticket']
titan_data.drop(columns=dp_list, inplace=True)

#Dropping duplicate values
titan_data = titan_data.drop_duplicates(keep='first')


#Convert 'Sex' to numeric
titan_data['Sex'] = titan_data['Sex'].map({'male': 0, 'female': 1})

#Splitting data for training and testing with a logical function
split = ssf(n_splits=1, test_size=0.2)
for train_ind, test_ind in split.split(titan_data, titan_data[["Survived", "Pclass", "Sex"]]):
    strain_set = titan_data.loc[train_ind] 
    stest_set = titan_data.loc[test_ind]
    
#Training X and Y data 
train_X = strain_set.drop(['Survived'], axis = 1)
train_Y = strain_set['Survived']

#Testing X and Y data
test_X = stest_set.drop(['Survived'], axis = 1)
test_Y = stest_set['Survived']


# Train the model with Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(train_X, train_Y)
rf_acc = accuracy_score(test_Y, model_rf.predict(test_X))

print("Random Forest Accuracy:", rf_acc)
# Save model
joblib.dump(model_rf, 'titanic_rf_model.pkl')


# Predict values
y_pred = model_rf.predict(test_X)

# Generate confusion matrix
cm = confusion_matrix(test_Y, y_pred)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])

plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()







