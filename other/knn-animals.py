from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dltoolkit.preprocess import ResizePreprocessor
from dltoolkit.load import DataLoader
from imutils import paths
import argparse

# Check script arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to data set")
ap.add_argument("-k", "--neighbours", type=int, default=1, help="# of NN")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs")
args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))

# Load data
proc = ResizePreprocessor(32, 32)
dl = DataLoader(preprocessors=[proc])
(X, Y) = dl.load(imagePaths, verbose = 500)
X = X.reshape(-1, 3072)

# Encode labels to int
lenc = LabelEncoder()
Y = lenc.fit_transform(Y)

# Split train/test set
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)

# Fit classifier on training set
knn = KNeighborsClassifier(n_neighbors=args["neighbours"], n_jobs=args["jobs"])
knn.fit(X_train, Y_train)

# Predict test set
print(classification_report(Y_test, knn.predict(X_test), target_names=lenc.classes_))
