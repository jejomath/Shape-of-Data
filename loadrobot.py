import math, numpy, scipy, random, matplotlib.pyplot
import sklearn.linear_model as lm

# Load the data as strings
print "Loading data..."
f = open("robotlp1.data")
lines = [l[:-1] for l in f]
f.close()

# Calculate the number of rows in the data matrix
rows = len(lines)/18+1

# Create an empty data matrix and labels list
data = scipy.zeros([rows, 90])
labels = []

# Cycle through each block of data
for i in range(rows):
    # Read the label from the first line of the block
    labels.append(lines[i*18])

    for j in range(15): # Cycle through the data lines
        d = lines[i*18 + j + 1].split('\t')
        # Read in the six numbers from each time step
        for k in range(6):
            data[i,j*6+k] = float(d[k+1])


# Show PCA plot of all four classes
colors = {'normal':'green','collision':'blue','obstruction':'red','fr_collision':'yellow'} # Dictionary taking classes to colors
M = (data-numpy.mean(data.T, axis=1)).T # Mean-center the data

[latent, coeff] = numpy.linalg.eig(numpy.cov(M)) # Find principal components
coords = numpy.dot(coeff.T, M)      # Convert the data to PC coordinates

fig, ax = matplotlib.pyplot.subplots() # Create matplotlib figure
for i in range(len(labels)): # Plot the first 500 points
    ax.scatter(coords[0,i], coords[1,i], color=colors[labels[i]])


# Show PCA of just normal and 'normal' and 'collision'

ncindices = [i for i in range(len(labels)) if labels[i] == 'normal' or labels[i] == 'collision']

ncdata = data[ncindices,:]
nclabels = [l for l in labels if l == 'normal' or l == 'collision']

M = (ncdata-numpy.mean(ncdata.T, axis=1)).T # Mean-center the data

[latent, coeff] = numpy.linalg.eig(numpy.cov(M)) # Find principal components
coords = numpy.dot(coeff.T, M)      # Convert the data to PC coordinates

fig, ax = matplotlib.pyplot.subplots() # Create matplotlib figure
for i in range(len(nclabels)): # Plot the first 500 points
    ax.scatter(coords[0,i], coords[1,i], color=colors[nclabels[i]])




