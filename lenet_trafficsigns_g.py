# USAGE
# python lenet_trafficsigns_g.py --training ./datasets/GTSRB/Training/ --model lenet_trafficsigns_weights_g.hdf5 --output output --weights weights/improvements
#find ./ -printf "%f\n" | sort

# import the necessary packages

import matplotlib
matplotlib.use("Agg") #ensures figures can be saved in backgroudn

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from cnn.preprocessing import ImageToArrayPreprocessor
from cnn.preprocessing import SimplePreprocessor
from cnn.datasets import SimpleDatasetLoader
from cnn.nn.conv import LeNet
from cnn.callbacks import TrainingMonitor

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help ="path to output model")
ap.add_argument("-o", "--output", required=True,
	help= "path to output dictionary")
ap.add_argument("-w", "--weights", required=True,
	help= "path to weights directory")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid())) #shows the process ID info

print("[INFO] loading images...")
trainingImagePaths = list(paths.list_images(args["training"]))

sp = SimplePreprocessor(28,28)
iap = ImageToArrayPreprocessor()

#list of preprocessors to be applied in sequential order. First reordered to 32x32, then channel ordered properly to keras.json file
sdl = SimpleDatasetLoader(preprocessors=[sp,iap]) #loads dataset then scale the raw pixel intensities from [0,1]
(data,labels) = sdl.load(trainingImagePaths,verbose = 500)
data = data.astype("float")/255.0 #scale to range [0,1]

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size = 0.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["00000", 
			  "00001",
			  "00002",
			  "00003",
			  "00004",
			  "00005",
			  "00006",
			  "00007",
			  "00008",
			  "00009",
			  "00010",
			  "00011",
			  "00012",
			  "00013",
			  "00014",
			  "00015",
			  "00016",
			  "00017",
			  "00018",
			  "00019",
			  "00020",
			  "00021",
			  "00022",
			  "00023",
			  "00024",
			  "00025",
			  "00026",
			  "00027",
			  "00028",
			  "00029",
			  "00030",
			  "00031",
			  "00032",
			  "00033",
			  "00034",
			  "00035",
			  "00036",
			  "00037",
			  "00038",
			  "00039",
			  "00040",
			  "00041",
			  "00042"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=3, classes=43)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


#training monitor
figPath = os.path.sep.join([args["output"], "{}.png".format(
	os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(
	os.getpid())])
#callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]


# construct the callback to save only the current *best* model to disk based on the validation loss
fname = os.path.sep.join([args["weights"],
	"weights-{epoch:03d}-{val_loss:.4f}.hdf5"]) #naming: epoch# to 3 digits, and second is variable set for monitoring
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
	save_best_only=True, verbose=1) #creates model checkpoint. Lower the minimum val_loss the better.
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), checkpoint] #chain callbacks


# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=128, epochs=40, callbacks=callbacks, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

#plot training loss and accuracy.
#commented out b/c monitoring path
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
