import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from parsing_vics import vics_media
import re
import PIL
import PIL.Image
from PIL import ImageFile
import secrets
import random
import subprocess


def main():
	has_flag_img = ['7c12427ce1d89418655c114db40bea64', \
	'8ce50fe438b8e0c208cd8dd66a252f9e',\
	'64398fd6850a0ddc26a3c91d933d1974',\
	'7aa51e8e605a4e7ab0c74da4c1b90d8f',\
	'86ccba6e1b6b372e20e7a718c287e80f',\
	'79cb3e7253a33bacab1f43ab0233379b',\
	'1c290534497ac1244856d5e7e76629a4',\
	'29200df76d67cccc0c05fac3ff140027',\
	'7428d7c0170ed627d009ef87e9358229',\
	'a342f4531499aaf2863dc19e6d4f71ab',\
	'd0f2dccfad844c57efa9c8a6ab06b8af',\
	'69a63cd3bd2de3f700efe19990505310',\
	'c75531b4bb0eb348147d2c1c4426cc85',\
	'b59bd67aa6afaeaf15384e3eab210be2',\
	'ba0a568f88835a78322c84a7c596418d',\
	'59281a1b302020e18761f144f87c23e3',\
	'7cde8b72a5a9183eb572d50f915705eb',\
	'cfc2cbb3db25e879d1ab1f60fd9b3670',\
	'34691f1833f046269634e9b385f34b02',\
	'1fbf919d6cc5c3a9c8d7780f7a894849',\
	'35c4ebe17c55e42c53faf9b53c2e61ea',\
	'91468771e0518201998ba624f1837602',\
	'889f2291f0978b3cd2fd5d72a9569bd8']


	has_flag_vid = ['9b49f6ceb3bd326b253bec2541d7dc6b', \
	'6a63f3192e57edf7b4b0c8ad27c9b2a8', \
	'dcbeb053b2f60b1934e06bb4c35d41f6', \
	'f5b2e69d36cc668bbe352e4919bfe331', \
	'729fdd854125fc92d751c5aa6610f82b', \
	'fdcdd12198760a0599319e8407cb85c8', \
	'2e40dc2263e4f08bd8e2e41a4633b99e', \
	'5c1c734656b23034c203a987bf1c5289', \
	'6ccb6b45a2e4669dd96b0ffdc055755e', \
	'39449aa13477aeb4b6c67a07a38a71aa', \
	'836f9e1f0f40fe7e94edb39ee726ddca', \
	'89e6034dcf7d552806f37e1aaedebd24', \
	'09086fde753c1cc16e12e3f0c670c5ab', \
	'f966a1c3b755bc2d847cdf5792385d79', \
	'151faf004e2f89eec532bcd97a821f0d', \
	'e191a4e7b8c81d7a45035ff0060c8342', \
	'5837f5f995296241a118e6ac1797a87c', \
	'01e06af3170c4ee786adf97aea466ff0', \
	'5e28c0f6b6b7af14d60247fbe7bf94eb', \
	'249e3678721d2b087054db581af871e4', \
	'542cab8a220b226124ad53c5628841fa', \
	'f649b2f0db6910c0966414edb8660dad', \
	'4107e38537723e6c6522c3fb8e4905ea']

#1. Load json file and create vics media objects for all media.
	json_file = open("C:\\Users\\Stuart\\OneDrive\\Documents from\
	 laptop\\Minor Thesis Forensic Files\\Exported_files\\VID_and_\
	 Image_file_types\\all_media_files_by_signature-raw-text.JSON", "r")
	media_files = vics_media.loading_vics_json(json_file)
	json_file.close()
	holder = vics_media()
	vics_media_list = []
	for file in media_files:
		file = str(file)
		holder = vics_media.str_to_vics(file)
		holder.clean_vics()
		vics_media_list.append(holder)

#2. Load spreadsheet using pandas. Get the extension column and filename 
#and zip them into a dictionary.
	column_data = pd.read_csv("C:\\Users\\Stuart\\OneDrive\\Documents from\
	 laptop\\Minor Thesis Forensic Files\\Exported_files\\all attachments \
	 EXIF data\\Autopsy metadata.csv",\
	 usecols =['MD5 Hash','Added_extension'])
	column_data = column_data.set_index('MD5 Hash')
	filetype_dict = column_data.to_dict()
	another_dict = filetype_dict['Added_extension']
	
#3 Look up each file in the dictionary and update its extension field.
	search_term = ""
	target_files =[]
	filetypes_set = set()
	for media in vics_media_list:
		searchterm = media.md5
		media.extension = another_dict[searchterm]
		if searchterm in has_flag_img or searchterm in has_flag_vid:
			media.target = True
		if media.target == True:
			target_files.append(media)
		filetypes_set.add(media.extension)
	#for target_media in target_files:
	#	print(target_media)
	#print(str(filetypes_set))

#4. Load all media data into a tf data frame.
#4-1. Open the image file. The format of the file can be JPEG, PNG, BMP, etc.
	count = 0
	count2 =0
	fail_count = 0
	img_media_processed = []
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	for media in vics_media_list:
		try:	
			if media.extension == 'webm' or media.extension == 'mp4' or media\
			.extension == 'db':
				count2 += 1
				pass
			else:
				file_path = \
				"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis \
				Forensic Files\\Exported_files\\VID_and_Image_file_types\\" \
				+ media.relative_file_path
				#image = np.array(PIL.Image.open(file_path))
				image = PIL.Image.open(file_path)
				count +=1
				height, width = image.size
				print(media.relative_file_path + "," + str(height) + "," + str(width))
				img_media_processed.append(media)
		except OSError:
			fail_count += 1
			pass

	print("\n\nThe Length of the vics_media_list is " + str(len(vics_media_list)) + \
		" files.\n\nThe count of webm, mp4 and db (thumbnail) files is: " + str(count2) \
		+ "\n\nThe count of files attempted opening is: " + str(count) \
		+ "\n\nThe number of files that could not be processed is: " + str(fail_count) \
		+ "\n\nThis adds to a total of " + str(count2 + count) + " files processed and "\
		 + str(fail_count) + " excluded.")

#4-2. Resize the image to match the input size for the Input layer of the Deep Learning model.
#Examination of file sizes suggests that approx. 80 % of all images are smaller than the mean average
#for both height and width. All files were resized to these averages in the initial testing.
	"""
	new_height = 256 #476
	new_width = 256 #354
	for media in vics_media_list:
		try:
			if media.extension == 'webm' or media.extension == 'mp4' or media.extension\
			 == 'db':
					count2 += 1
					pass
			else:	
				file_path = \
				"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic\
				 Files\\Exported_files\\VID_and_Image_file_types\\"\
				+ media.relative_file_path
				image = PIL.Image.open(file_path)
				image = image.resize((new_width,new_height))
				image.save("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
				 Forensic Files\\Exported_files\\VID_and_Image_file_types\\resized2\\" + str(\
				 media.relative_file_path), format=media.extension)
		except OSError:
			pass
	"""

#4.25 create three datasets. 13190 files can be processed. Three equal groups for training, 
#testing and validation of 4396 files. This section will then not need to be run again.
	"""
	flags = has_flag_img
	processed_images = img_media_processed
	counter = 0
	training = []
	testing = []
	validation = []
	holder = 0
	while flags != []:
		hash_file = secrets.choice(flags)
		if counter % 3 == 0:
			training.append(hash_file)
			flags.remove(hash_file)
			counter += 1
		elif counter % 3 == 1:
			testing.append(hash_file)
			flags.remove(hash_file)
			counter += 1
		else: 
			validation.append(hash_file)
			flags.remove(hash_file)
			counter += 1

	print("training: " + str(training))
	print("testing: " + str(testing))
	print("validation: " + str(validation))
#Replace each hash with its media file equivilent and remove the media file from the list \
#available to be allocated.
	tg_holder = []
	for tg in training:
		for proc_media in processed_images:
			if tg == proc_media.md5:
				tg_holder.append(proc_media)
				processed_images.remove(proc_media)
				training.remove(tg)
	training = tg_holder

	test_holder = []
	for test in testing:
		for proc_media in processed_images:
			if test == proc_media.md5:
				test_holder.append(proc_media)
				processed_images.remove(proc_media)
				testing.remove(test)
	testing = test_holder

	validation_holder = []
	for vl in validation:
		for proc_media in processed_images:
			if vl == proc_media.md5:
				validation_holder.append(proc_media)
				processed_images.remove(proc_media)
				validation.remove(vl)
	validation = validation_holder


	print("training: " + str(training))
	print("testing: " + str(testing))
	print("validation: " + str(validation))

	counter = 0
	while processed_images != []:
		media_file = secrets.choice(processed_images)
		if counter % 3 == 0:
			training.append(media_file)
			processed_images.remove(media_file)
			counter += 1
		elif counter % 3 == 1:
			testing.append(media_file)
			processed_images.remove(media_file)
			counter += 1
		else:
			validation.append(media_file)
			processed_images.remove(media_file)
			counter += 1

	random.shuffle(training)
	print(training)
	random.shuffle(testing)
	print(testing)
	random.shuffle(validation)
	print(validation)

#All files should now be distributed into one of the three datasets. Write them into files. 
	training_dataset = open("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
	 Forensic Files\\datasets\\training.csv","w+")
	training_dataset.write("Relative Path,md5,extension\n")
	for tg in training:
		training_dataset.write(str(tg.relative_file_path) + "," + str(tg.md5) + "," + str(tg.\
		extension) + "\n")
	training_dataset.close()

	testing_dataset = open("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis \
	Forensic Files\\datasets\\testing.csv","w+")
	testing_dataset.write("Relative Path,md5,extension\n")
	for test in testing:
		testing_dataset.write(str(test.relative_file_path)  + "," + str(test.md5)  + "," + \
		str(test.extension) + "\n")
	testing_dataset.close()

	validation_dataset = open("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
	 Forensic Files\\datasets\\validation.csv","w+")
	validation_dataset.write("Relative Path,md5,extension\n")
	for vl in validation:
		validation_dataset.write(str(vl.relative_file_path)  + "," + str(vl.md5)  + "," + str(vl\
		.extension) + "\n")
	validation_dataset.close()
	"""

# Move files into folder for processing (Training Data).
	"""
	processed_files_path = "C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
	 Forensic Files\\Exported_files\\VID_and_Image_file_types\\resized"
	training_handle = open("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
	 Forensic Files\\datasets\\training.csv","r")
	training_data = training_handle.readlines()
	training_handle.close()
	training_data = training_data[1:]
	training_file = []
	training_hashes = []
	for item in training_data:
		item = item.split(',')
		training_file.append(item[0])
		training_hashes.append(item[1])

	command =""
	training_targets = []
	for img_hash in has_flag_img:
		if img_hash in training_hashes:
			training_targets.append(img_hash)

	for target_hash in training_targets:
		for vics in vics_media_list:
			if target_hash == vics.md5:
				img_file = vics
				cleaned = img_file.relative_file_path.replace('\\\\','\\',1)
				img_file_path = "\"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
				 Forensic Files\\Exported_files\\VID_and_Image_file_types\\resized2\\" + cleaned
				command = "copy " + img_file_path + "\" \"C:\\Users\\Stuart\\Documents\\training\\\
				target\\" + target_hash + "." + vics.extension + "\""
				print(command)
				subprocess.call(command,shell=True)

	for vics in vics_media_list:
		print(vics.md5)
		if vics.md5 in training_hashes:
			img_file = vics
			cleaned = img_file.relative_file_path.replace('\\\\','\\',1)
			img_file_path = "\"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
			 Forensic Files\\Exported_files\\VID_and_Image_file_types\\resized2\\" + img_file.\
			 relative_file_path
			command = "copy " + img_file_path + "\" \"C:\\Users\\Stuart\\Documents\\training\\\
			ignorable\\" + vics.md5 + "." + vics.extension + "\""
			print(command)
			subprocess.call(command,shell=True)

	"""
	#"copy C:\\Users\\stewi\\Documents\\Minor Thesis Forensic Files\\Exported_files\\VID_and_\
	#Image_file_types\\resized2"
	
#Move files into folder for processing (Validation Data)

	"""	
	processed_files_path = "C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic \
	Files\\Exported_files\\VID_and_Image_file_types\\resized"
	validation_handle = open("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic \
	Files\\datasets\\validation.csv","r")
	validation_data = validation_handle.readlines()
	validation_handle.close()
	validation_data = validation_data[1:]
	validation_file = []
	validation_hashes = []
	for item in validation_data:
		item = item.split(',')
		validation_file.append(item[0])
		validation_hashes.append(item[1])

	command =""
	validation_targets = []
	for img_hash in has_flag_img:
		if img_hash in validation_hashes:
			validation_targets.append(img_hash)

	for target_hash in validation_targets:
		for vics in vics_media_list:
			if target_hash == vics.md5:
				img_file = vics
				cleaned = img_file.relative_file_path.replace('\\\\','\\',1)
				img_file_path = "\"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis \
				Forensic Files\\Exported_files\\VID_and_Image_file_types\\resized2\\" + cleaned
				command = "copy " + img_file_path + "\" \"C:\\Users\\Stuart\\Documents\\validation\\\
				target\\" + target_hash + "." + vics.extension + "\""
				print(command)
				subprocess.call(command,shell=True)

	for vics in vics_media_list:
		print(vics.md5)
		if vics.md5 in validation_hashes:
			img_file = vics
			cleaned = img_file.relative_file_path.replace('\\\\','\\',1)
			img_file_path = "\"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic\
			 Files\\Exported_files\\VID_and_Image_file_types\\resized2\\" + img_file.relative_file_path
			command = "copy " + img_file_path + "\" \"C:\\Users\\Stuart\\Documents\\validation\\ignorable\
			\\" + vics.md5 + "." + vics.extension + "\""
			print(command)
			subprocess.call(command,shell=True)
	"""

#Move files into folder for processing (Testing Data)
	"""
	processed_files_path = "C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic\
	 Files\\Exported_files\\VID_and_Image_file_types\\resized"
	testing_handle = open("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic\
	 Files\\datasets\\testing.csv","r")
	testing_data = testing_handle.readlines()
	testing_handle.close()
	testing_data = testing_data[1:]
	testing_file = []
	testing_hashes = []
	for item in testing_data:
		item = item.split(',')
		testing_file.append(item[0])
		testing_hashes.append(item[1])

	command =""
	testing_targets = []
	for img_hash in has_flag_img:
		if img_hash in testing_hashes:
			testing_targets.append(img_hash)

	for target_hash in testing_targets:
		for vics in vics_media_list:
			if target_hash == vics.md5:
				img_file = vics
				cleaned = img_file.relative_file_path.replace('\\\\','\\',1)
				img_file_path = "\"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis\
				 Forensic Files\\Exported_files\\VID_and_Image_file_types\\resized2\\" + cleaned
				command = "copy " + img_file_path + "\" \"C:\\Users\\Stuart\\Documents\\testing\\"\
				 + target_hash + "." + vics.extension + "\""
				print(command)
				subprocess.call(command,shell=True)

	for vics in vics_media_list:
		print(vics.md5)
		if vics.md5 in testing_hashes:
			img_file = vics
			cleaned = img_file.relative_file_path.replace('\\\\','\\',1)
			img_file_path = "\"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis \
			Forensic Files\\Exported_files\\VID_and_Image_file_types\\resized2\\" + img_file.\
			relative_file_path
			command = "copy " + img_file_path + "\" \"C:\\Users\\Stuart\\Documents\\testing\\" +\
			 vics.md5 + "." + vics.extension + "\""
			print(command)
			subprocess.call(command,shell=True)
	"""

#Attempt training
	"""
	
	img_height = 256
	img_width = 256
	batch_size = 32

	train_ds = tf.keras.utils.image_dataset_from_directory("C:\\Users\\Stuart\\Documents\\training\\all", \
 	labels = "inferred", \
 	#subset="training", \
	image_size=(img_height, img_width), \
	batch_size=batch_size)

	val_ds = tf.keras.utils.image_dataset_from_directory("C:\\Users\\Stuart\\Documents\\validation", \
 	labels = "inferred", \
 	#subset="validation", \
	image_size=(img_height, img_width), \
	batch_size=batch_size)


	class_names = train_ds.class_names
	print(class_names)


#Data visualisation
	im_squeeze = ""
	plt.figure(figsize=(10,10))
	for images, labels in train_ds.take(1):
		print(str(type(images)))
		print(str(images.shape))
		#type is EagerTensor
		for i in range(9):
			ax = plt.subplot(3,3,i+1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(class_names[labels[i]])
			plt.axis("off")
	plt.show()

#4-4. Normalize the image to have pixel values scaled down between 0 and 1 from 0 to 255.
	
	#Declare the normalisation layer rule
	normalisation_layer = tf.keras.layers.Rescaling(1./255)

	#From tensorflow tutorial: https://www.tensorflow.org/tutorials/load_data/images
	normalised_train_ds = train_ds.map(lambda x, y: (normalisation_layer(x), y))
	image_batch, labels_batch = next(iter(normalised_train_ds))


	normalised_val_ds = val_ds.map(lambda x, y: (normalisation_layer(x), y))
	image_batch, labels_batch = next(iter(normalised_val_ds))


#Configure the dataset for performance
	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)


#4-5 Create the model
	
	num_classes = 2
	model = tf.keras.Sequential([ \
	layers.Conv2D(16, 3, padding='same', activation='relu'), \
	layers.MaxPooling2D(), \
	layers.Conv2D(32, 3, padding='same', activation='relu'), \
	layers.MaxPooling2D(), \
	layers.Conv2D(64, 3, padding='same', activation='relu'), \
	layers.MaxPooling2D(), \
	layers.Flatten(), \
	layers.Dense(128, activation='relu'), \
	layers.Dense(num_classes)])

	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy\
	(from_logits=True), metrics=['accuracy'])

#Train the model
	epochs = 10
	history = model.fit(train_ds,validation_data=val_ds, epochs=epochs)
	model.save("C:\\Users\\Stuart\\Documents\\Image_classification\\saved_model1\\larger_training_set2")


#output a figure of accurance and performance for training and validation. 
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	
	epochs_range = range(epochs)
	
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')
	
	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()
	
	"""
if __name__ == '__main__':
	main()
