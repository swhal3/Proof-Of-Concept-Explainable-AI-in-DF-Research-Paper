import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import numpy
from tensorflow import keras
from tensorflow.keras import layers
from lime import lime_image
import time
from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.preprocessing import LabelEncoder
import PIL
import random
import numpy as np

def main():
	new_model = tf.keras.models.load_model("C:\\Users\\Stuart\\Documents\\Image_classification\\saved_model1\\larger_training_set2", compile=True)
	img_height = 256
	img_width = 256
	data_labels = ['ignorable','target']


	#Get names of all files that were tested.
	#2. Load spreadsheet using pandas. Get the extension column and filename and zip them into a dictionary.
	file_handle = open("C:\\Users\\Stuart\\Documents\\February_2022\\SavedModel1-results.txt", "r")
	results = file_handle.readlines()
	results = results[1:]
	file_handle.close()

	#split into data sets.
	true_pos= []
	true_neg= []
	false_pos= []
	false_neg= []
	
	for file in results:
		file = file.strip(" \n")
		split = file.split("\t")
		if split[-1] == "ignorable" and split[-2] == "ignorable":
			true_neg.append(file)
		elif split[-1] == "target" and split[-2] == "target":
			true_pos.append(file)
		elif split[-1] == "ignorable" and split[-2] == "target":
			false_pos.append(file)
		else:
			false_neg.append(file)

	print("The length of each list is:\n" +\
		"True Negative: " + str(len(true_neg)) + "\n" + \
		"True Positive: " + str(len(true_pos)) + "\n" + \
		"False Positive: " + str(len(false_pos)) + "\n" + \
		"False Negative: " + str(len(false_neg)))
	
	#Pick random samples of 50 files for each of true positive, true negative, false positive and false negative.
	sample_true_pos= random.sample(true_pos,50)
	sample_true_neg= random.sample(true_neg,50)
	sample_false_pos= random.sample(false_pos,50)
	sample_false_neg= random.sample(false_neg,50)

	#Process using LIME and save to a file.
	
	#Open each file
	name = ""
	testing_file_lists = [sample_true_pos,sample_true_neg,sample_false_pos,sample_false_neg]
	for samples in testing_file_lists:
		if samples == sample_true_pos:
			name = "sample_true_pos"
		elif samples == sample_true_neg:
			name = "sample_true_neg"
		elif samples == sample_false_pos:
			name = "sample_false_pos"
		else:
			name = "sample_false_neg"
		file_handle = open("C:\\Users\\Stuart\\Documents\\Lime_processed\\" + name + ".csv", "w+")
		file_handle.write("file,prediction,score,predicted_label\n")

		for pos in samples:
			file_handle.write(pos)
			pos = pos.strip(" \n")
			split = pos.split("\t")
			img = tf.keras.utils.load_img("C:\\Users\\Stuart\\Documents\\testing\\" + split[0], target_size=(img_height,img_width))
			lime_img_array = tf.keras.utils.img_to_array(img)
			predict_img_array = tf.expand_dims(lime_img_array, 0)
			explainer = lime_image.LimeImageExplainer(random_state=42)
			explanation = explainer.explain_instance(lime_img_array, new_model.predict, top_labels = 2, hide_color = 0, num_samples =1000, random_seed = 42)
			ig_image, ig_mask = explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=0)[0],positive_only=True,hide_rest=False, num_features=5)
			tg_image, tg_mask = explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=1)[0],positive_only=True,hide_rest=False, num_features=5)
	
			#Create the image
			fig = plt.figure(figsize=(10, 15))
			# setting values to rows and column variables
			rows = 3
			columns = 2
	
			fig.add_subplot(rows, columns, 1)
			plt.imshow(lime_img_array.astype('uint8'))
			plt.axis('off')
			plt.title("Original")
	
	
			fig.add_subplot(rows, columns, 2)
			plt.imshow(mark_boundaries(ig_image.astype('uint8'), ig_mask, mode='inner'))
			plt.axis('off')
			plt.title("Fields indicating ignorable")
	
			fig.add_subplot(rows,columns,3)
			plt.imshow(mark_boundaries(tg_image.astype('uint8'), tg_mask, mode='inner'))
			plt.axis('off')
			plt.title("Fields indicating Target")
	
			fig.add_subplot(rows,columns,5)
			plt.text(0, 1, "Image Name: " + split[0] + "\n"\
				+ "Expected Result: " + split[-1] + "\n"\
				+ "Predicted Result: " + split[-2] + "\n"
				+ "Confidence Scores: " + split[1] + " [[ignorable, target]]",\
				verticalalignment='top', horizontalalignment='left' )
			plt.axis('off')
	
			plt.title("Results")
	
	
			fig.add_subplot(rows,columns,4)
	
			#Get locations for target files
			ind = explanation.top_labels[1]
	
			dict_heatmap = dict(explanation.local_exp[ind])
			heatmap = np.vectorize(dict_heatmap.get) (explanation.segments)
	
			plt.imshow(heatmap, cmap = 'RdBu' , vmin = -heatmap.max(), vmax = heatmap.max())
			plt.colorbar()
			plt.axis('off')
			plt.title("Heatmap of fields")
	
			fig.savefig("C:\\Users\\Stuart\\Documents\\Lime_processed_reattempted\\" + name + "\\" + split[0].replace('.','_') + ".pdf")
			plt.close()
	
	
			
		file_handle.close()




	"""
	img = tf.keras.utils.load_img("C:\\Users\\Stuart\\Documents\\testing\\test_87aab450fb7adeb4fbcb4bab80a8df3ec.jpg", target_size=(img_height,img_width))
	LIME_img_array = tf.keras.utils.img_to_array(img)
	img = PIL.Image.open("C:\\Users\\Stuart\\Documents\\testing\\test_87aab450fb7adeb4fbcb4bab80a8df3ec.jpg")
	print(LIME_img_array.shape) #Shape here is (256, 256, 3) -> required for LIME
	predict_img_array = tf.expand_dims(LIME_img_array, 0)
	#print(predict_img_array.shape) #Shape is now (1,256,256,3) -> required for model prediction
	prediction = new_model.predict(predict_img_array) #.argmax(axis=1)[0]
	argmax_prediction = new_model.predict(predict_img_array).argmax(axis=1)
	argmax_prediction2 = new_model.predict(predict_img_array).argmax(axis=0)
	print(str(prediction))
	print(str(argmax_prediction))
	print(str(argmax_prediction2))

	#Set the explainer
	explainer = lime_image.LimeImageExplainer(random_state=42)

	#Set explanation
	explanation = explainer.explain_instance(LIME_img_array, new_model.predict, top_labels = 2, hide_color = 0, num_samples =1000, random_seed = 42)

	#Get image and mask

	#image, mask = explanation.get_image_and_mask(img_array, positive_only = True, hide_rest = False)
	image, mask = explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=0)[0],positive_only=True,hide_rest=False, num_features=2)
	

	plt.imshow(mark_boundaries(image.astype('uint8'), mask))
	plt.show()



	#Output image with original, boundaries indicating ignorable, boundaries indicating target, expected label and actual label
	#Actual Image
	plt.imshow(img)
	plt.show()
	
	#Prediction Image
	plt.imshow(tf.squeeze(predict_img_array))
	plt.show()
	
	print(type(mask))
	# Showing the lime image array - It's not the actual image
	plt.imshow(LIME_img_array.astype('uint8'))
	plt.show()
	plt.imshow(mask)
	plt.show()
	plt.show()
	"""



if __name__ == '__main__':
	main()
