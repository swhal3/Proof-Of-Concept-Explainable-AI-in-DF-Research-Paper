#Running LIME Metadata processing using text classification model

#Import modules
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
from tensorflow.keras import layers
from tensorflow.keras import losses
import pandas as pd
import numpy as np
import lime
from lime import lime_text
import shap

#This function is a modified version of the same function in the Text Classifier tutorial on the tensorflow website
#It undertakes transformations on the input data to make the predictions better.
def custom_standardization(input_data):
	lowercase = tf.strings.lower(input_data)
	formating = tf.strings.regex_replace(input_data, '\\', ' ')
	formating = tf.strings.regex_replace(formating, '  ', ' ')
	formating = tf.strings.strip(input_data)
	return formating


"""The original function - For testing
def custom_standardization(input_data):
	lowercase = tf.strings.lower(input_data)
	stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
	return tf.strings.regex_replace(stripped_html,\
	'[%s]' % re.escape(string.punctuation),\
	'')
"""


def main():
	#First, all the records in the CSV needed to be processed into appropriate folders to be made into a dataset.

	#Read the CSV
	training = pd.read_csv("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\training_sample_metadata_fields_removed.csv")
	#print(training.info())
	
	#Convert Written, Created and Accessed to Dates so target data can be labelled
	training['Created'] = training['Created'].astype('datetime64[ns]')
	training['Accessed'] = training['Accessed'].astype('datetime64[ns]')
	training['Written'] = training['Written'].astype('datetime64[ns]')

	target_training_data = training[~training["FilePath"].str.contains("OneDrive")]
	target_training_data = target_training_data[(target_training_data["Created"] > "2021-09-04 00:00:00") & (target_training_data['Created'] < "2021-09-04 12:00:00")]
	vhds_of_interest = ['image15','image16','image17','image18','image4','image5','image6',]
	target_training_data = target_training_data[target_training_data["FilePath"].str.contains("image15") | target_training_data["FilePath"].str.contains("image16") \
	| target_training_data["FilePath"].str.contains("image17") | \
	target_training_data["FilePath"].str.contains("image18") | \
	target_training_data["FilePath"].str.contains("image4") | \
	target_training_data["FilePath"].str.contains("image5") | \
	target_training_data["FilePath"].str.contains("image6")]	
	target_training_data['Prediction'] = "target"

	#Convert all the target indexes to a list
	target_indexes = target_training_data.index.values.tolist()
	
	#Write in the label to the dataframe and mark them in the training dataset.
	count = 0
	prediction_column = []
	for x in range(0,len(training)):
		if x in target_indexes:
			prediction_column.append('target') #is target
		else:
			prediction_column.append('ignorable') #is ignorable
	training['Prediction'] = prediction_column
	#training['Created'] = training['Created'].astype('int64')
	#training['Accessed'] = training['Accessed'].astype('int64')
	#training['Written'] = training['Written'].astype('int64')



	#Training is now a pandas dataframe with a prediction column with all the target entries tagged in the prediction column
	
	"""
	#HERE, we write each file into its corresponding label folder

	training_list = training.values.tolist()
	file_counter = 0

	for item in training_list:

		#Save Target files
		if item[-1] == 'target':
			file_handle = open("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_training_data\\target\\"\
			 + str(file_counter) + ".txt", "w+")
			file_handle.write(str(item[0:len(item)-2]))
			file_handle.close()
			file_counter+=1

		#Save Ignorable files
		else:
			file_handle = open("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_training_data\\ignorable\\"\
			 + str(file_counter) + ".txt", "w+")
			file_handle.write(str(item[0:len(item)-2]))
			file_handle.close()
			file_counter+=1
	

	#Create Testing Dataset

	testing = pd.read_csv("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\testing_sample_data_fields_removed.csv")
	testing = testing.values.tolist()
	file_counter = 0
	for item in testing:
		file_handle = open("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_testing_data\\test\\"\
		 + str(file_counter) + ".txt", "w+")
		file_handle.write(str(item[0:len(item)-2]))
		file_handle.close()
		file_counter+=1

	"""


	#Constructing the model following the Tensorflow text classification tutorial

	batch_size = 1
	seed = 42

	raw_training_dataset = tf.keras.utils.text_dataset_from_directory("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_training_data\\",\
	labels='inferred',shuffle=True, seed=seed, batch_size = batch_size, validation_split = 0.2, subset = 'training')

	raw_validation_dataset = tf.keras.utils.text_dataset_from_directory("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_training_data\\",\
	shuffle=True, seed=seed, batch_size = batch_size, validation_split = 0.2, subset = 'validation')

	raw_test_dataset = tf.keras.utils.text_dataset_from_directory("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_testing_data\\",\
	 batch_size=batch_size)

	#Controls how many words to add to the index
	max_features = 10000

	sequence_length = 250

	#As per the tutorial
	vectorize_layer = layers.TextVectorization(standardize=custom_standardization,\
	max_tokens = max_features,\
	output_mode = 'int',
	output_sequence_length = sequence_length)


	#As per the tutorial
	def vectorize_text(text, label):
		text = tf.expand_dims(text, -1)
		return vectorize_layer(text), label


	train_text = raw_training_dataset.map(lambda x, y: x)
	vectorize_layer.adapt(train_text)

	train_dataset = raw_training_dataset.map(vectorize_text)
	print(train_dataset)
	val_dataset = raw_validation_dataset.map(vectorize_text)
	print(val_dataset)
	test_dataset = raw_test_dataset.map(vectorize_text)
	print(test_dataset)

	embedding_dim = 16
	model = tf.keras.Sequential([\
	layers.Embedding(max_features + 1, embedding_dim),\
	layers.Dropout(0.2),\
	layers.GlobalAveragePooling1D(),\
	layers.Dropout(0.2),\
	layers.Dense(2)])

	class_names = ['ignorable', 'target']
	model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),\
	optimizer='adam',\
	metrics=['accuracy'])

	epochs = 10

	history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

	print(model.summary())

	#model.save("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_classifier_model\\model")

	#Export model with text vectorization layer. However, I haven't been able to load it because it needs the Text Vectorisation
	#Encapsulated in it, and I don't know how to do that. i just retrain it each time I run a test.

	export_model = tf.keras.Sequential([\
	vectorize_layer,
	model,
	layers.Activation('sigmoid')])

	export_model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

	#loss, accuracy = export_model.evaluate(raw_test_dataset)

	
	#export_model.save("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_classifier_model\\model_with_textvectorisation_v2")
	
	#Running Predictions.

	testing = pd.read_csv("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\testing_sample_data_fields_removed.csv")

	#export_model = tf.keras.models.load_model("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\text_classifier_model\\model_with_textvectorisation_v2")
	count = 0
	test_list = testing.values.tolist()
	for item in test_list[0:10]:
		item = str(item)
		predict = export_model.predict([item])
		#prob = tf.nn.sigmoid(predict[0])
		print(item + " : " + str(predict))
		explainer = lime_text.LimeTextExplainer(class_names=class_names)
		exp = explainer.explain_instance(item, export_model.predict, num_features = 6)
		print(exp.available_labels())
		html = exp.show_in_notebook(text=False)
		file_handle = open("C:\\Users\\Stuart\\Documents\\METADATA_PROECSSING_v2\\results\\" + str(count) + ".html", "w+")
		file_handle.write(html)
		file_handle.close()
		count += 1


	"""
	Predictions run fine. However I am unable to create the explainer object in LIME. I believe this is because LIME requires
	export_model.predict([item]) to return an numpy array with two values x and y, which are the probability scores for the input
	being a ignorable and a target respectively. x + y should equal 1.

	This is similar to the output of the model.predict_proba() method in scikit-learn, which is not implemented in tensorflow.

	I have to modify the model from the tutorial to fix this. 
	

	The Error encountered is:

	Traceback (most recent call last):
	File "text_classifier_V1.py", line 244, in <module>
	main()
	File "text_classifier_V1.py", line 213, in main
	exp = explainer.explain_instance(item, export_model.predict, num_features = 6)
	lime_text.py", line 432, in explain_instance
	lime_base.py", line 182, in explain_instance_with_data
	labels_column = neighborhood_labels[:, label]
	IndexError: index 1 is out of bounds for axis 1 with size 1
	"""










	#FOR TESTING - NOT OF INTEREST

	"""
	explainer = lime_text.LimeTextExplainer()
	exp = explainer.explain_instance([an_item], export_model.predict, num_features=5,labels=(1,))
	"""

	#trying Shap instead
	"""
	explainer = shap.DeepExplainer(instance, export_model.predict, train_dataset[:100])

	shap_values = explainer.shap_values(test_dataset[:10])

	shap.initjs()

	shap.force_plot(explainer.expected_value[0], shap_values[0][0])
	"""





if __name__ == '__main__':
	main()