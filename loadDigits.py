import scipy, random, math
from Tkinter import *
from sklearn.cluster import KMeans

# The mode variable determines which algorithms are run and
# which images are displayed. The modes are described below.
# To modify the behavior of each mode (such as changing the number
# of centroids), scroll down to the end of the script.

### Mode 0: Draw 25 images of each digit
### Mode 1: Draw average letters
### Mode 2: Run K-means on each digit
### Mode 3: Take averages of consecutive images
### Mode 4: Run K-means on the whole set

mode = 2


# This class defines the window that displays rows of grey-scale letters.
# The size of the images and the number of columns to be displayed is
# controlled by the varibles at the end of the __init__ function.
# The script sends letters to the window via the drawLetter function.

class Application(Frame):
    def __init__(self, master=None):
        self.master = master
        self.master.title("Shape of Data Digit Explorer")

        self.frame = Frame(master)
        self.frame.pack(padx=5, pady=5)
        self.master.protocol("WM_DELETE_WINDOW", self.frame.quit)

        # The size of the window is determined by the side of the canvas.
        self.drawC = Canvas(master, width=690, height=300)
        self.drawC.pack()

        self.psize = 3      # The size of individual pixels in each image
        self.charwidth = 8  # The width of each images
        self.charheight = 8 # The height of each image
        self.ncols = 25     # The number of images to draw in each row
        self.curcol = 0     # The column where the next image should be drawn
        self.currow = 0     # The row where the next image should be drawn
 
    def drawLetter(self, pixels):
        ps = self.psize # shorter version of pixel size

        # The x- and y-values of the upper left corner of the new image
        osx = 10 + (self.charwidth+1)*self.psize*self.curcol
        osy = 10 + (self.charheight+1)*self.psize*self.currow

        # Draw the image
        for i in range(self.charwidth):
            for j in range(self.charheight):
                c = 255 - (pixels[i*self.charwidth+j]*255.0/16)
                a = "#" + '%02x%02x%02x' % (int(c), int(c), int(c))
                self.drawC.create_rectangle([osx+j*ps,osy+i*ps,osx+(j+1)*ps-1,osy+(i+1)*ps-1], fill=a, outline=a)

        # Update the current column and row
        self.curcol += 1
        if self.curcol >= self.ncols:
            self.curcol = 0
            self.currow += 1



# Load the digits file and convert to data and labels
data = scipy.loadtxt("optdigits.tra", delimiter=',')
labels = data[:,-1]
data = data[:,:-1]

# Create the display window.
root = Tk()
app = Application(master=root)

### Mode 0: Draw 25 images of each digit
if mode == 0:
    for i in range(10):
        ind = [j for j in range(labels.shape[0]) if labels[j] == i]
        for j in range(25):
            app.drawLetter(data[ind[j]])

### Mode 1: Draw average letters
elif mode == 1:
    for i in range(10):
        ind = [j for j in range(labels.shape[0]) if labels[j] == i]
        avgletter = scipy.zeros([64])
        for j in range(64):
            avgletter[j] = float(sum([data[k,j] for k in ind]))/len(ind)
        app.drawLetter(avgletter)

### Mode 2: Run K-means on each digit
elif mode == 2:
    ncents = 25 # Number of centroids for each digit    
    app.ncols = ncents
    
    for i in range(10):
        ind = [j for j in range(labels.shape[0]) if labels[j] == i]
        KM = KMeans(init='random', n_clusters=ncents)
        KM.fit(data[ind])

        for j in range(ncents):
            app.drawLetter(KM.cluster_centers_[j])


### Mode 3: Take averages of consecutive images
elif mode == 3:
    for i in range(50):
        avgletter = scipy.zeros([64])
        for j in range(64):
            avgletter[j] = float(sum([data[k,j] for k in range(i*30, (i+1)*30)]))/30
        app.drawLetter(avgletter)

### Mode 4: Run K-means on the whole set
elif mode == 4:
    ncl = 50

    KM = KMeans(init='random', n_clusters=ncl)
    KM.fit(data)

    for i in range(ncl):
        app.drawLetter(KM.cluster_centers_[i])

# Final code needed to display the window.
root.mainloop()
root.destroy()

