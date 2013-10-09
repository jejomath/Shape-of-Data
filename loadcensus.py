import math, numpy, scipy, random, matplotlib.pyplot
import sklearn.linear_model as lm

# Tokens list used to create binary features
tokens = [[],
          ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
          [],
          ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
          [],
          ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
          ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
          ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
          ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
          ['Female', 'Male'],
          [],
          [],
          [],
          ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']]

# Count where each category starts
fstart = []
count = 0
for f in tokens:
    fstart.append(count)
    count += len(f)


# Load the data as strings
print "Loading data..."
f = open("census.data")
lines = [l for l in f]  # REad in all the lines
lines = lines[:-1]      # Remove the final empty line
f.close()

# Create the empty data matrix
data = scipy.zeros([len(lines), count])

# Populate data matrix using the tokens
for i in range(len(lines)):     # For each line of data
    l = lines[i].split(',')     # Split the string into tokens
    for j in range(len(l)-1):   # For each category
        if l[j][1:] in tokens[j]:   # If the token is valid
            # Calculate what feature it defines
            k = fstart[j] + tokens[j].index(l[j][1:])
            data[i,k] = 1       # Set the feature to 1

# Create the empty labels vector
labels = scipy.zeros(len(lines))

# Convert the labels to numbers
for i in range(len(lines)):     # For each line of data
    l = lines[i].split(',')     # Split into tokens
    # Read the first character of the label
    if l[-1][1] == '>':
        labels[i] = 1.0
    else:
        labels[i] = 0.0

# Optionally, scale the data and try PCA again
# To have the script scale the data, change the 
# first line to "if 1 == 1:"
if 1 == 0:
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