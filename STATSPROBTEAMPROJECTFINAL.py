#fit an ANN model
#data set source URL: https://www.openml.org/search?type=data&status=active&id=552&sort=runs
import keras.models
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import array

col_names = ['id','FTP','UEMP','MAN','LIC','GR','CLEAR','WM','NMAN','GOV','HE','WE','HOM','ACC','ASR']
# load dataset
detroit = pd.read_csv("csv_result-detroit.csv", header=None, names=col_names)

detroit.head()

# Preprocessing
detroit = detroit.apply(pd.to_numeric, errors='coerce')
detroit.fillna(detroit.mean(), inplace=True)

feature_cols = ['FTP','UEMP','MAN','LIC','GR','CLEAR','WM','NMAN','GOV','HE','WE','ACC','ASR']
X = detroit[feature_cols].values

# Calculate the average value of the "hom" column
average_hom = detroit['HOM'].mean()
print(f"Average homicides: {average_hom}")

# Filter rows where "hom" > average
greater_than_average = detroit[detroit['HOM'] > average_hom]

detroit['hom_above_average'] = detroit['HOM'].apply(lambda x: 1 if x > average_hom else 0)

y = detroit['hom_above_average'].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# build a model

model = Sequential()
model.add(Dense(16, input_shape=(X.shape[1],), activation='relu')) # Add an input shape! (features,)
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary() 

# compile the model
model.compile(optimizer='Adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# early stopping callback
# This callback will stop the training when there is no improvement in  
# the validation loss for 10 consecutive epochs.  
es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max', # don't minimize the accuracy!
                                   patience=10,
                                   restore_best_weights=True)

# now we just update our model fit call
# Train ANN
history = model.fit(X, y, 
                    batch_size=10, 
                    epochs=80, 
                    validation_split=0.2, 
                    verbose=1)
#Evaluate the Model


history_dict=history.history
#learning curve(Loss)
#lets see the training and validation loss by epoch

#loss
loss_values= history_dict['loss'] #you can change this
val_loss_values = history_dict['val_loss'] #you can also change this

#range of X(no. of epochs)
epochs = range(1,len(loss_values)+1)

#plot
import matplotlib.pyplot as plt

plt.plot(epochs, loss_values,'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# orange is for "orange"
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# this is the max value - should correspond to
# the HIGHEST train accuracy
np.max(val_acc)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# see how these are numbers between 0 and 1? 
model.predict(X) # prob of successes (survival)
np.round(model.predict(X),0) # 1 and 0 (survival or not)
y # 1 and 0 (survival or not)







# so we need to round to a whole number (0 or 1),
# or the confusion matrix won't work!
preds = np.round(model.predict(X),0)

# confusion matrix

cm = confusion_matrix(y, preds) # order matters! (actual, predicted)





#plot the confusion matrix

import seaborn as sns

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('True Label');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

## Display the visualization of the Confusion Matrix.
plt.show()





# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y, preds))
print("Precision:",metrics.precision_score(y, preds))
print("Recall:",metrics.recall_score(y, preds))
print("F1 score:",metrics.f1_score(y, preds))



