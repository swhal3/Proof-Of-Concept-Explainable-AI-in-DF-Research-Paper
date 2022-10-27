import subprocess
import tensorflow as tf
from tensorflow import keras
from parsing_vics import vics_media
import os
import PIL
import PIL.Image
from PIL import ImageFile
import matplotlib.pyplot as plt
matplotlib.use('pgf')
import matplotlib
import numpy as np
import pandas as pd
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.preprocessing import LabelEncoder

def main():
	"""
	#Find dataset - Same as media data loader, but we are only collecting video files.
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

	"""
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

	"""

#1. Load json file and create vics media objects for all media.
	json_file = open("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic Files\\Exported_files\\VID_and_Image_file_types\\all_media_files_by_signature-raw-text.JSON", "r")
	media_files = vics_media.loading_vics_json(json_file)
	json_file.close()
	holder = vics_media()
	vics_media_list = []
	for file in media_files:
		file = str(file)
		holder = vics_media.str_to_vics(file)
		holder.clean_vics()
		vics_media_list.append(holder)

#2. Load spreadsheet using pandas. Get the extension column and filename and zip them into a dictionary.
	column_data = pd.read_csv("C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic Files\\Exported_files\\all attachments EXIF data\\Autopsy metadata.csv",\
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
	video_files = []
	for media in vics_media_list:
		if media.extension == 'webm' or media.extension == 'mp4':
			file_path = \
			"C:\\Users\\Stuart\\OneDrive\\Documents from laptop\\Minor Thesis Forensic Files\\Exported_files\\VID_and_Image_file_types\\" \
			+ media.relative_file_path
			video_files.append(file_path)
		else:
			pass

	for video in video_files:
		print(video)
	
	#Process video into key frames
	video_counter = 1
	for video in video_files:

		get_file_name = video.split("\\")
		file_name = get_file_name[-1].replace('.','_')
		print("Processing video " + str(video_counter) + " of " + (str(len(video_files))) + " : " + file_name)
		video_counter += 1
		vc = cv2.VideoCapture(video)
		frame_counter = 1
		if vc.isOpened():
			rval , frame = vc.read()
		else:
			rval = False

		while rval:
			if frame_counter % 25 ==  0:
				rval, frame = vc.read()
				try:
					cv2.imwrite('C:\\Users\\Stuart\\Documents\\Video_Frames\\' + file_name + "_" + str(frame_counter) + '.jpg',frame)
					frame_counter += 1
					cv2.waitKey(1)
				except cv2.error:
					print(file_name + " Errored. Check if it processed at all.")
					break
			else:
				rval, frame = vc.read()
				frame_counter +=1
				pass
	"""
	#Resize and Process keyframes through image classification model
	img_height = 256
	img_width = 256
	class_names = ['ignorable','target']
	"""
	#Get the video frames from the folder.

	frames = os.listdir("C:\\Users\\Stuart\\Documents\\Video_Frames\\")

	for frame in frames:
		try:	
			file_path = "C:\\Users\\Stuart\\Documents\\Video_Frames\\" + frame
			image = PIL.Image.open(file_path)
			image = image.resize((img_width,img_height))
			image.save("C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\" + str(frame), format='JPEG')
		except OSError:
			pass
	"""
	"""
	#PROCESS videos through model.
	new_model = tf.keras.models.load_model("C:\\Users\\Stuart\\Documents\\Image_classification\\saved_model1\\larger_training_set2", compile=True)
	resized_frames = os.listdir("C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\")
	resized_frames.sort()
	prediction = ""
	results = open("C:\\Users\\Stuart\\Documents\\Video_Results\\Video_results1.tsv", "w+")
	results.write("file\tprediction\tscore\tpredicted_label\tderived_score\n")

	derived_score = 0
	score_list = []
	video_name = ""
	previous_frame_name = ""
	frame_num = ""
	frames = []
	for resized_frame in resized_frames:
		file_path = "C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\" + resized_frame
		img = tf.keras.utils.load_img(file_path, target_size=(img_height,img_width))
		img_array = tf.keras.utils.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0)
		prediction = new_model.predict(img_array)
		#predict_label = prediction.argmax(axis=-1)
		predict_label = class_names[np.argmax(prediction)]
		score = tf.nn.softmax(prediction[0])
		derived_score = np.sum(prediction)
		results.write(resized_frame + "\t" + str(prediction) + "\t" + str(score) + "\t" + str(predict_label) + "\t" + str(derived_score) +"\n")
		print(resized_frame + "\t" + str(prediction) + "\t" + str(score) + "\t" + str(derived_score) +"\n")
	results.close()
	"""
	#Aggregate results of key frames to individual prediction for video
	column_data = pd.read_excel("C:\\Users\\Stuart\\Documents\\Video_Results\\Video_results1.xlsx",\
	sheet_name = 'Video_results_with_frame_info',\
	usecols=['file','Video','frame','derived_score','ignorable','target','predicted_label','Ignorable_Rounded','Target_Rounded'])
	
	get_video_names = column_data['Video'].unique().tolist()
	data = ""

	new_model = tf.keras.models.load_model("C:\\Users\\Stuart\\Documents\\Image_classification\\saved_model1\\larger_training_set2", compile=True)

	get_video_names = ['f5b2e69d36cc668bbe352e4919bfe331']
	for video in get_video_names:
		is_target = False
		if video in has_flag_vid:
			is_target = True
		data = column_data[column_data.Video == video]
		y = data.target
		x = data.frame
		z = data.ignorable
		#a = data.derived_score
		ig_rounded = data.Ignorable_Rounded
		ig_rounded = ig_rounded*-1
		tg_rounded = data.Target_Rounded
		prediction = data.mode(axis=0)
		prediction = prediction.iat[0,4]
		
		#get target files for analysis
		score_column = data['target']
		tg_max_index = score_column.idxmax()
		tg_min_index = score_column.idxmin()
		tg_max_score_file = data['file'][tg_max_index]
		tg_min_score_file = data['file'][tg_min_index]


		#get ignorable files for analysis scoring file
		score_column = data['ignorable']
		ig_max_index = score_column.idxmax()
		ig_min_index = score_column.idxmin()
		ig_max_score_file = data['file'][ig_max_index]
		ig_min_score_file = data['file'][ig_min_index]

		min_max_explan = "Maximum Target Score Frame: " + str(data['frame'][tg_max_index]) + "\n" + \
		"Minimum Target Score Frame: " + str(data['frame'][tg_min_index]) + "\n" + \
		"Maximum Ignorable Score Frame: " + str(data['frame'][ig_max_index]) + "\n" + \
		"Minimum Ignorable Score Frame: " + str(data['frame'][ig_max_index]) + "\n\n" + \
		"This video was predicted as: " + prediction + "\n\n"


		#analyse max target frame in LIME - The frame which had the highest score for being a target
		img = tf.keras.utils.load_img("C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\" + tg_max_score_file, target_size=(img_height,img_width))
		lime_img_array = tf.keras.utils.img_to_array(img)
		predict_img_array = tf.expand_dims(lime_img_array, 0)
		explainer = lime_image.LimeImageExplainer(random_state=42)
		tg_max_explanation = explainer.explain_instance(lime_img_array, new_model.predict, top_labels = 2, hide_color = 0, num_samples =1000, random_seed = 42)
		tg_max_ig_image, tg_max_ig_mask = tg_max_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=0)[0],positive_only=True,hide_rest=False, num_features=5)
		tg_max_tg_image, tg_max_tg_mask = tg_max_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=1)[0],positive_only=True,hide_rest=False, num_features=5)

		#analyse min target frame in LIME - The frame that had the lowest score for being a target
		img = tf.keras.utils.load_img("C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\" + tg_min_score_file, target_size=(img_height,img_width))
		lime_img_array = tf.keras.utils.img_to_array(img)
		predict_img_array = tf.expand_dims(lime_img_array, 0)
		explainer = lime_image.LimeImageExplainer(random_state=42)
		tg_min_explanation = explainer.explain_instance(lime_img_array, new_model.predict, top_labels = 2, hide_color = 0, num_samples =1000, random_seed = 42)
		tg_min_ig_image, tg_min_ig_mask = tg_min_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=0)[0],positive_only=True,hide_rest=False, num_features=5)
		tg_min_tg_image, tg_min_tg_mask = tg_min_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=1)[0],positive_only=True,hide_rest=False, num_features=5)

		#analyse max ignorable frame in LIME - The frame that had the highest score for being ignorable
		img = tf.keras.utils.load_img("C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\" + ig_max_score_file, target_size=(img_height,img_width))
		lime_img_array = tf.keras.utils.img_to_array(img)
		predict_img_array = tf.expand_dims(lime_img_array, 0)
		explainer = lime_image.LimeImageExplainer(random_state=42)
		ig_max_explanation = explainer.explain_instance(lime_img_array, new_model.predict, top_labels = 2, hide_color = 0, num_samples =1000, random_seed = 42)
		ig_max_ig_image, ig_max_ig_mask = ig_max_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=0)[0],positive_only=True,hide_rest=False, num_features=5)
		ig_max_tg_image, ig_max_tg_mask = ig_max_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=1)[0],positive_only=True,hide_rest=False, num_features=5)

		#analyse max ignorable frame in LIME - The frame that had the lowest score for being ignorable
		img = tf.keras.utils.load_img("C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\" + ig_min_score_file, target_size=(img_height,img_width))
		lime_img_array = tf.keras.utils.img_to_array(img)
		predict_img_array = tf.expand_dims(lime_img_array, 0)
		explainer = lime_image.LimeImageExplainer(random_state=42)
		ig_min_explanation = explainer.explain_instance(lime_img_array, new_model.predict, top_labels = 2, hide_color = 0, num_samples =1000, random_seed = 42)
		ig_min_ig_image, ig_min_ig_mask = ig_min_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=0)[0],positive_only=True,hide_rest=False, num_features=5)
		ig_min_tg_image, ig_min_tg_mask = ig_min_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=1)[0],positive_only=True,hide_rest=False, num_features=5)

		
		"""
		#analyse Lowest scoring frame in LIME
		img = tf.keras.utils.load_img("C:\\Users\\Stuart\\Documents\\Video_Frames_resized\\" + lowest_score_file, target_size=(img_height,img_width))
		lime_img_array = tf.keras.utils.img_to_array(img)
		predict_img_array = tf.expand_dims(lime_img_array, 0)
		explainer = lime_image.LimeImageExplainer(random_state=42)
		low_explanation = explainer.explain_instance(lime_img_array, new_model.predict, top_labels = 2, hide_color = 0, num_samples =1000, random_seed = 42)
		low_ig_image, low_ig_mask = low_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=0)[0],positive_only=True,hide_rest=False, num_features=5)
		low_tg_image, low_tg_mask = low_explanation.get_image_and_mask(new_model.predict(predict_img_array).argmax(axis=1)[0],positive_only=True,hide_rest=False, num_features=5)
		"""
		
		fig = plt.figure(figsize=(10,15))
		rows = 5
		columns = 2
		"""
		#fig.add_subplot(rows, columns,1)
		"""

		#TARGET FRAMES
		fig.add_subplot(rows, columns, 1)
		plt.title("Highest Scoring Target Frame: " + str(data['frame'][tg_max_index]) + ": Ignorable Fields",fontsize=10)
		plt.imshow(mark_boundaries(tg_max_ig_image.astype('uint8'), tg_max_ig_mask, mode='inner'))
		plt.axis('off')

		fig.add_subplot(rows, columns, 2)
		plt.title("Highest Scoring Target Frame: " + str(data['frame'][tg_max_index]) + ": Target Fields",fontsize=10)
		plt.imshow(mark_boundaries(tg_max_tg_image.astype('uint8'), tg_max_tg_mask, mode='inner'))
		plt.axis('off')

		fig.add_subplot(rows, columns, 3)
		plt.title("Lowest Scoring Target Frame: " + str(data['frame'][tg_min_index]) + ": Ignorable Fields",fontsize=10)
		plt.imshow(mark_boundaries(tg_min_ig_image.astype('uint8'), tg_min_ig_mask, mode='inner'))
		plt.axis('off')

		fig.add_subplot(rows, columns, 4)
		plt.title("Lowest Scoring Target Frame: " + str(data['frame'][tg_min_index]) + ": Target Fields",fontsize=10)
		plt.imshow(mark_boundaries(tg_min_tg_image.astype('uint8'), tg_min_tg_mask, mode='inner'))
		plt.axis('off')

		#IGNORABLE FRAMES

		fig.add_subplot(rows, columns, 5)
		plt.title("Highest Scoring Ignorable Frame: " + str(data['frame'][ig_max_index]) + ": Ignorable Fields",fontsize=10)
		plt.imshow(mark_boundaries(ig_max_ig_image.astype('uint8'), ig_max_ig_mask, mode='inner'))
		plt.axis('off')

		fig.add_subplot(rows, columns, 6)
		plt.title("Highest Scoring Ignorable Frame: " + str(data['frame'][ig_max_index]) + ": Target Fields",fontsize=10)
		plt.imshow(mark_boundaries(ig_max_tg_image.astype('uint8'), ig_max_tg_mask, mode='inner'))
		plt.axis('off')

		fig.add_subplot(rows, columns, 7)
		plt.title("Lowest Scoring Ignorable Frame: " + str(data['frame'][ig_min_index]) + ": Ignorable Fields",fontsize=10)
		plt.imshow(mark_boundaries(ig_min_ig_image.astype('uint8'), ig_min_ig_mask, mode='inner'))
		plt.axis('off')

		fig.add_subplot(rows, columns, 8)
		plt.title("Lowest Scoring Ignorable Frame: " + str(data['frame'][ig_min_index]) + ": Target Fields",fontsize=10)
		plt.imshow(mark_boundaries(ig_min_tg_image.astype('uint8'), ig_min_tg_mask, mode='inner'))
		plt.axis('off')

		fig.add_subplot(rows, columns, 9)
		plt.text(0,0, min_max_explan)
		plt.axis('off')
		#plt.show()
		plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
		"""
		#Sort Results
		if is_target:
			if prediction == 'target':
				fig.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\true_pos\\" + video + "_Keyframe_LIME.pdf")
				plt.close()
			else:
				fig.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\false_neg\\" + video + "_Keyframe_LIME.pdf")
				plt.close()
		else:
			if prediction == 'ignorable':
				fig.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\true_neg\\" + video + "_Keyframe_LIME.pdf")
				plt.close()
			else:
				fig.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\false_pos\\" + video + "_Keyframe_LIME.pdf")
				plt.close()				
		"""

		fig2 = plt.figure(figsize=(20,15))
		plt.scatter(x,y, c='b')
		plt.scatter(x,z, c='r')
		#plt.scatter(x,a, c='g')
		plt.xlabel("Frame", fontsize=14)
		plt.ylabel("Predicted Score",fontsize=14)
		plt.axhline(y=0, color='r', linestyle='-')
		plt.vlines(x=x, ymin=ig_rounded, ymax = tg_rounded, color='g')
		plt.title(video + ": Changes in Prediction Scores over Frames")
		plt.plot(x,y, label='target')
		plt.plot(x,z, label='ignorable')
		#plt.plot(x,a, label='Derived Score (target + ignorable)')
		plt.legend(fontsize = 14)
		fig2.savefig("C:\\Users\\Stuart\\OneDrive\\Desktop\\image.eps")
		fig2.savefig("C:\\Users\\Stuart\\OneDrive\\Desktop\\image.pdf")
		fig2.savefig("C:\\Users\\Stuart\\OneDrive\\Desktop\\image.pgf")
		plt.close()


		"""
		#Sort Results
		if is_target:
			if prediction == 'target':
				fig2.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\true_pos\\" + video + "_graphed_results.pdf",orientation='landscape')
			#	fig2.savefig("C:\\Users\\Stuart\\Documents\\test.pdf",orientation='landscape')				
				plt.close()
			else:
				fig2.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\false_neg\\" + video + "_graphed_results.pdf",orientation='landscape')
			#	fig2.savefig("C:\\Users\\Stuart\\Documents\\test.pdf",orientation='landscape')				
				plt.close()
		else:
			if prediction == 'ignorable':
				fig2.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\true_neg\\" + video + "_graphed_results.pdf",orientation='landscape')
			#	fig2.savefig("C:\\Users\\Stuart\\Documents\\test.pdf",orientation='landscape')			
				plt.close()
			else:
				fig2.savefig("C:\\Users\\Stuart\\Documents\\Video_keyFrames_output\\false_pos\\" + video + "_graphed_results.pdf",orientation='landscape')
			#	fig2.savefig("C:\\Users\\Stuart\\Documents\\test.pdf",orientation='landscape')

				plt.close()

		"""
		"""
		
		fig.add_subplot(rows,columns,6)
	
		#Get locations for target files
		ind = high_explanation.top_labels[1]
	
		dict_heatmap = dict(high_explanation.local_exp[ind])
		high_heatmap = np.vectorize(dict_heatmap.get) (high_explanation.segments)
	
		plt.imshow(high_heatmap, cmap = 'RdBu' , vmin = -high_heatmap.max(), vmax = high_heatmap.max())
		plt.colorbar()

		plt.title("Heatmap of fields for Highest Scoring Frame")

		fig.add_subplot(rows,columns,7)
	
		#Get locations for target files
		ind = low_explanation.top_labels[1]
	
		dict_heatmap = dict(low_explanation.local_exp[ind])
		low_heatmap = np.vectorize(dict_heatmap.get) (low_explanation.segments)
	
		plt.imshow(low_heatmap, cmap = 'RdBu' , vmin = -low_heatmap.max(), vmax = low_heatmap.max())
		plt.colorbar()

		plt.title("Heatmap of fields for Lowest Scoring Frame")
		plt.show()

		#pick out most target and most ignorable - process through lime and graph output.
		
		"""

if __name__ == '__main__':
	main()

