import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('Phishing_Legitimate.csv') ## Change here based on where is file on your local
df = df.dropna()
print(df.shape)

df = df.dropna()
X = df.drop(['id','CLASS_LABEL'], axis=1).values
Y = df[['CLASS_LABEL']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, stratify=Y, random_state=123)
print('Train/Test Sizes : ',X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print('Build model...')
model = Sequential([
                    LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(1,48)),
                    Dense(1, activation='sigmoid')
                   ])
print(model.summary())

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train.reshape((X_train.shape[0],1,X_train.shape[1])), Y_train, batch_size=32, epochs=7, validation_split=0.1)


loss, acc = model.evaluate(X_test.reshape((X_test.shape[0],1,X_test.shape[1])), Y_test, batch_size=64)
preds = model.predict_classes(X_test.reshape((X_test.shape[0],1,X_test.shape[1])))

print('Test Accuracy : ', accuracy_score(Y_test, preds))
print('Test Loss:', loss)
print('Test accuracy:', acc)


print('Build model...')
dense_model = Sequential( [Dense(256, activation='relu', input_shape=(48,)),
                           Dense(128, activation='relu'),
                           Dense(64,activation='relu'),
                           Dense(1, activation='sigmoid')]
                  )
print(dense_model.summary())

# try using different optimizers and different optimizer configs
dense_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = dense_model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_split=0.1)

loss, acc = dense_model.evaluate(X_test, Y_test, batch_size=64)
preds = dense_model.predict_classes(X_test)

print('Test Accuracy : ', accuracy_score(Y_test, preds))
print('Test Loss:', loss)
print('Test accuracy:', acc)
