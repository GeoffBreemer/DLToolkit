"""Animal classification using k-NN (three classes)"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dltoolkit.preprocess import ResizePreprocessor
from dltoolkit.io import DataLoader
from imutils import paths
import argparse

# Check script arguments
ap = argparse.ArgumentParser(description="Apply k-NN to animal images.")
ap.add_argument("-d", "--dataset", required=True, help="path to the data set")
ap.add_argument("-k", "--neighbours", type=int, default=1, help="# of nearest neighbours to use")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of sciki-learn jobs to use")
args = vars(ap.parse_args())

# Extract the full path to each image in the data set's location
imagePaths = list(paths.list_images(args["dataset"]))

# Load the images, resizing each to 32x32 pixels upon loading
proc = ResizePreprocessor(32, 32)
dl = DataLoader(preprocessors=[proc])
(X, Y) = dl.load(imagePaths, verbose=500)

# Reshape the features from (# of records, 32, 32, 3) to (# of records, 32*32*3=3072)
X = X.reshape(-1, 3072)

# Encode the labels from strings to integers
lenc = LabelEncoder()
Y = lenc.fit_transform(Y)

# Split into a training and test set
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)

# Fit a k-NN classifier on training set
knn = KNeighborsClassifier(n_neighbors=args["neighbours"], n_jobs=args["jobs"])
knn.fit(X_train, Y_train)

# Make predictions on the test set and print the results to the console
print(classification_report(Y_test, knn.predict(X_test), target_names=lenc.classes_))
