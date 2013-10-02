import numpy, matplotlib.pyplot
import sklearn.linear_model
import sklearn.svm


# labelnames is a dictionary that translates the species names into integers
labelnames = {"Iris-setosa\n":0, "Iris-versicolor\n":1, "Iris-virginica\n":2}
data, labels = [], []      # Create empty lists for the vectors and labels.
f = open("iris.data", "r") # Open the data file.
for line in f:             # Load in each line from the file. 
    vals = line.split(',') # Split the line into individual values
    if len(vals) == 5:     # Check that there are five columns
        data.append(vals[:-1])              # Add the data vector
        labels.append(labelnames[vals[4]])  # Add the numerical label
f.close()                  # Close the file



colors = ['red','green','blue'] # Dictionary taking classes to colors
data = numpy.array(data, dtype="float") # Convert the data to numpy array
M = (data-numpy.mean(data.T, axis=1)).T # Mean-center the data

[latent, coeff] = numpy.linalg.eig(numpy.cov(M)) # Find principal components
coords = numpy.dot(coeff.T, M)      # Convert the data to PC coordinates

fig, ax = matplotlib.pyplot.subplots() # Create matplotlib figure
for i in range(data.shape[0]): # Plot all the points
    ax.scatter(coords[0,i], coords[1,i], color=colors[labels[i]])




# Create the Logistic Regression object
LR = sklearn.linear_model.LogisticRegression()

LR.fit(data[50:], labels[50:]) # Find the hyperplane

pred = LR.predict(data[50:])   # Predict the classes

# Count the number of correct predictions
correct = len([i for i in range(100) if pred[i] == labels[i+50]])

print "Logistic regression correctly precicts", correct, 
print "out of 100."



# Create the Logistic Regression object
LR = sklearn.svm.SVC()

LR.fit(data[50:], labels[50:]) # Find the hyperplane

pred = LR.predict(data[50:])   # Predict the classes

# Count the number of correct predictions
correct = len([i for i in range(100) if pred[i] == labels[i+50]])

print "The support vector machine correctly precicts", correct, 
print "out of 100."

