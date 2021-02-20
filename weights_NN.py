import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

dataframe = pandas.read_csv("cat_weights_csv.csv", header=None)
data = dataframe.values

x = data[:,0].astype(float)
y = data[:,1]

# encoder = LabelEncoder()
# encoder.fit(x)
# encoded_x = encoder.transform(x)


dummy_y = np_utils.to_categorical(y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=1, activation='relu'))
	model.add(Dense(5, activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=32, verbose=0)
