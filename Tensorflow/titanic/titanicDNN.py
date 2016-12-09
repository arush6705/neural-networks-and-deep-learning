import numpy as np
import tflearn

"""Upgrade tflearn if needed pip install -U git+https://github.com/tflearn/tflearn.git"""
# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

"""The Dataset is stored in a csv file, so we can use TFLearn load_csv() function to load the data from file into a python list. We specify 'target_column' argument to indicate that our labels (survived or not) are located in the first column (id: 0). The function will return a tuple: (data, labels)."""
# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)

"""First, we will discard the fields that are not likely to help in our analysis. For example, we make the assumption that 'name' field will not be very useful in our task, because we estimate that a passenger name and his chance of surviving are not correlated. With such thinking, we discard 'name' and 'ticket' fields.

Then, we need to convert all our data to numerical values, because a neural network model can only perform operations over numbers. However, our dataset contains some non numerical values, such as 'name' or 'sex'. Because 'name' is discarded, we just need to handle 'sex' field. In this simple case, we will just assign '0' to males and '1' to females."""
# Preprocessing function
def preprocess(data, columns_to_ignore):
	for eachPoint in range(len(data)):
		data[eachPoint] = [data[eachPoint][eachVar] for eachVar in range(len(data[eachPoint])) if eachVar not in columns_to_ignore]
	for i in range(len(data)):
		# Converting 'sex' field to float (id is 1 after removing labels column)
		data[i][1] = 1. if data[i][1] == 'female' else 0.
	return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)
"""We are building a 3-layers neural network using TFLearn. We need to specify the shape of our input data. In our case, each sample has a total of 6 features and we will process samples per batch to save memory, so our data input shape is [None, 6] ('None' stands for an unknown dimension, so we can change the total number of samples that are processed in a batch)."""
# # Build neural network
# None for no.of samples per batch
# 6 for no.of.variables
net = tflearn.input_data(shape=[None, 6])
# 32 neurons
net = tflearn.fully_connected(net, 32)
# 32 neurons
net = tflearn.fully_connected(net, 32)
# 2 output neurons, activation function = softmax
net = tflearn.fully_connected(net, 2, activation='softmax')
# fit a classic regression
net = tflearn.regression(net)
"""TFLearn provides a model wrapper 'DNN' that can automatically performs a neural network classifier tasks, such as training, prediction, save/restore, etc... We will run it for 10 epochs (the network will see all data 10 times) with a batch size of 16. """
# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

"""let's take Titanic movie protagonists (DiCaprio and Winslet) and calculate their chance of surviving (class 1)."""
# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])


