import math, numpy, scipy, random, matplotlib.pyplot, string
import sklearn.linear_model as lm
from collections import Counter
import sys

f = open("SMSSpamCollection.data", "r")
# Read each line of the data file, removing punctuation
lines = [l.translate(string.maketrans("",""), string.punctuation).lower() for l in f]
f.close()

wordcount = Counter() # Tracks how often each word appears

for l in lines:             # For each text message
    for w in l.split()[1:]: # For each word in the text
        wordcount[w] += 1   # Record it in wordcount

# Create a list of words that appear at least five times
words = [w for w in wordcount if wordcount[w] > 5]

# Create empty data and label holders
data = scipy.zeros([len(lines),len(words)])
labels = scipy.zeros(len(lines))

for i in range(len(lines)):     # For each text message
    l = lines[i].split()        # Make a list of words
    for j in range(len(words)): # For each token
        if words[j] in l[1:]:   # Set the entry to 1.0 if
            data[i,j] = 1.0     # the word appears

    if l[0] == "spam":   # Set the labels
        labels[i] = 1.0

# Optionally, scale the data and try PCA again
# To have the script scale the data, change the 
# first line to "if 1 == 1:"
if 1 == 1:
    s = [1.0/math.pow(sum(data[:,i]),.5) for i in range(data.shape[1])]
    data = data.dot(scipy.diag(s))

# Show PCA plot
colors = ['green','blue'] # Dictionary taking classes to colors
M = (data-numpy.mean(data.T, axis=1)).T # Mean-center the data

[latent, coeff] = numpy.linalg.eig(numpy.cov(M)) # Find principal components
coords = numpy.dot(coeff.T, M)      # Convert the data to PC coordinates

fig, ax = matplotlib.pyplot.subplots() # Create matplotlib figure
for i in range(500): # Plot the first 500 points
    ax.scatter(coords[0,i], coords[1,i], color=colors[int(labels[i])])

# Split the data into train and eval sets
print "Splitting into train an eval..."

# Pick out which indices correspond to which classes
A = [i for i in range(labels.shape[0]) if labels[i] == 1]
B = [i for i in range(labels.shape[0]) if labels[i] == 0]

# Randomize the sets of indices
random.shuffle(A)
random.shuffle(B)

# Calculate how many points to put in the training set
tA = int(len(A) * 0.8)
tB = int(len(B) * 0.8)

# Compile the lists of indices for the two sets
ti = A[:tA] + B[:tB]
ei = A[tA:] + B[tB:]

# Create the traing and test sets
tdata = data[ti]
tlabels = labels[ti]

# Create the labels lists for each
edata = data[ei]
elabels = labels[ei]

# Run logistic regression
LR = lm.LogisticRegression()

print "Running logistic regression..."

LR.fit(tdata, tlabels)

pred = LR.predict(edata)

# Calculate and report the accuracy numbers
correct = len([i for i in range(elabels.shape[0]) if pred[i] == elabels[i]])
total = elabels.shape[0]

print "Predicted", correct, "out of", total, 
print "correctly for a score of", float(correct)/total