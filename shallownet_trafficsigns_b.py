# USAGE
# python shallownet_trafficsigns_b.py --training ./datasets/BelgiumTSC_Training --testing ./datasets/BelgiumTSC_Testing --model shallownet_trafficsigns_weights_b.hdf5

# import the necessary packages

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from shallownet.preprocessing import ImageToArrayPreprocessor
from shallownet.preprocessing import SimplePreprocessor
from shallownet.datasets import SimpleDatasetLoader
from shallownet.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-tr", "--training", required=True,
	help="path to input training dataset")
ap.add_argument("-te", "--testing", required=True, 
	help="path to input testing dataset")
ap.add_argument("-m", "--model", required=True,
	help ="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading images...")
trainingImagePaths = list(paths.list_images(args["training"]))
testingImagePaths = list(paths.list_images(args["testing"]))

sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

#list of preprocessors to be applied in sequential order. First reordered to 32x32, then channel ordered properly to keras.json file
sdl = SimpleDatasetLoader(preprocessors=[sp,iap]) #loads dataset then scale the raw pixel intensities from [0,1]
(trainX,trainY) = sdl.load(trainingImagePaths,verbose = 500)
trainX = trainX.astype("float")/255.0

(testX,testY) = sdl.load(testingImagePaths,verbose = 500)
testX = testX.astype("float")/255.0

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
			  "00042",
			  "00043",
			  "00044",
			  "00045",
			  "00046",
			  "00047",
			  "00048",
			  "00049",
			  "00050",
			  "00051",
			  "00052",
			  "00053",
			  "00054",
			  "00055",
			  "00056",
			  "00057",
			  "00058",
			  "00059",
			  "00060",
			  "00061"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=62)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=40, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

#plot training loss and accuracy
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
