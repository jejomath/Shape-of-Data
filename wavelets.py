import scipy
from sklearn.cluster import KMeans


def findToken(token, series):
    dot_prod = []
    for i in range(series.shape[0] - token.shape[0]):
        dot_prod.append(sum([series[i+j] * token[j]
                          for j in range(token.shape[0])]))
    return max(dot_prod)


def makeVocab(data, length):
    vocab = []
    for d in data:
        for i in range(d.shape[0] - length + 1):
            vocab.append(d[i:i + length])
    return scipy.array(vocab)


def shortenVocab(long_vocab, n_tokens):
    KM = KMeans(init='random', n_clusters=n_tokens)
    KM.fit(long_vocab)
    return KM.cluster_centers_


def transformToVectors(data, length, n_tokens):
    long_vocab = makeVocab(data, length)
    short_vocab = shortenVocab(long_vocab, n_tokens)

    vector_data = scipy.zeros([len(data), n_tokens])

    for i in range(len(data)):
        for j in range(n_tokens):
            vector_data[i,j] = findToken(short_vocab[j], data[i])

    return vector_data

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

print "Transforming data..."

tdata = transformToVectors(data, 30, 50)
