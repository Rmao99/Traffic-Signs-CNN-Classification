# python trafficsigns_load.py --testing ./datasets/GTSRB/Testing/ --model lenet_trafficsigns_weights_g.hdf5

# import the necessary packages
from cnn.preprocessing import ImageToArrayPreprocessor
from cnn.preprocessing import SimplePreprocessor
from cnn.datasets import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testing", required=True,
	help="path to testing dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["00000", 
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

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["testing"])))
idxs = np.random.randint(0, len(imagePaths), size=(15,)) #loads 10 random images to be tested
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(28, 28)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=128).argmax(axis=1)

# loop over the sample images
d=0
for (i, imagePath) in enumerate(imagePaths):
	# load the example image, draw the prediction, and display it
	# to our screen
	image = cv2.imread(imagePath)
	cv2.putText(image, "{}".format(classLabels[preds[i]]),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.imwrite("img_%d.jpg"%d,image)
	cv2.waitKey(0)
	d+=1
