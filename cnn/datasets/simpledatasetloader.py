# import the necessary packages
import numpy as np
import cv2
import os

import pandas as pd

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images

		'''old code:
		for (i, imagePath) in enumerate(imagePaths):
			
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]
		'''		
		for csvPath in imagePaths:
			parts = csvPath.split('/')
			directory = parts[0]
			#csvFile = parts[1]
			
			csv = pd.read_csv(csvPath, sep=';')
			generator = csv.iterrows() #returns a generator

			for (i,row) in enumerate(list(generator)):
				image = cv2.imread(directory + '/' + row[1]['Filename'])
				#label = directory
				label = [directory, int(row[1]['Roi.X1']), int(row[1]['Roi.Y1']), int(row[1]['Roi.X2']),int(row[1]['Roi.Y2']) ]

		
			
				#test = pd.read_csv('GT-00000.csv',sep=';')
				#image = cv2.imread(test['Filename'][i]
				#label = [test['ClassId'][i],test['Roi.X1'][i],test['Roi.Y1'][i],test['Roi.X2'][i],test['Roi.Y2'][i]]
			#end of new code
				# check to see if our preprocessors are not None
				if self.preprocessors is not None:
					# loop over the preprocessors and apply each to
					# the image
					for p in self.preprocessors:
						image = p.preprocess(image)

				# treat our processed image as a "feature vector"
				# by updating the data list followed by the labels
				data.append(image)
				labels.append(label)

				# show an update every `verbose` images
				if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
					print("[INFO] processed {}/{}".format(i + 1,
						len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))
