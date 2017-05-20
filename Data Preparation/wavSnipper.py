from pydub import AudioSegment
import os

class WavSnipper:

	def __init__(self, segment_length, file_directory, store_directory):
		# specifies how large segments should be
		self.segment_length = segment_length
		# specifies the directory in which to look for audio files
		# if it is not the current directory, we need an additional slash in
		# order to reference it
		if file_directory == ".":
			self.file_directory = file_directory
		else:
			self.file_directory = "/{}".format(file_directory)
		# specifies the directory in which to store .wav-files
		self.store_directory = store_directory
		# create a list in which the file names of all files/subdirectories
		# are stored as strings
		self.file_list = os.listdir(self.file_directory)
		# enumeration variable for the expoted files
		self.enumerate = 0


	def start_snipping(self):
		'''
		starts an automatic snipping process which snips all .wav-files
		contained in the specified file directory into segments of specified
		length and stores them in the specified store directory
		'''
		# check if subdirectory exists
		if not os.path.exists("{}/{}".format(self.file_directory,
											 self.store_directory)):
			# if not the case, create it
			os.makedirs("{}/{}".format(self.file_directory,
									   self.store_directory))
			# iterate through all elements contained in the directory
			for file_name in self.file_list:
				# get file type of element
				file_type = file_name.split(".")[-1]
				# check if .wav-file
				if file_type == "wav":
					# and call the following function to split the audio file
					try:
						self.split_into_segments(
							file_name=file_name, export_name="musicdata")
					# fetch exception if it is a directory, not an audio file
					except IsADirectoryError:
						print("{} is a directory, not an audio file".format(
							file_name))
				# if not .wav-file, say so
				else:
					print("File {} not supported, not a .wav-file".format(
						file_name))
		else:
			print("Subdirectory {} already exists. Delete it if you want me to\
			 do anything, otherwise I will just do nothing".format(
				self.store_directory))


	def split_into_segments(self, file_name, export_name):
		'''
		splits an audio file specified by string "file_name" into segments of
		specified length and stores them in individual .wav-files (used by
		function start_snipping())
		'''
		sound = AudioSegment.from_wav(file_name)
		# first get the duration of the entire audio file in seconds
		duration = sound.duration_seconds
		# then get the number of segments we need:
		# for an uneven number of segments, we simply drop the last, unfitting part
		# of the audio file for ease of use, hence conversion to integer
		number_of_segments = int(duration / self.segment_length)
		# and specify the first segment's beginning and end
		segment_start = 0
		segment_end = self.segment_length*1000
		# in this for-loop, we export each segment that is defined by segment_start
		# and segment_end as a seperate .wav-file, then move to the next segment
		for i in range(number_of_segments):
			# we cut the sound at the corresponding position and save it to a new variable
			sound_export = sound[segment_start:segment_end]
			# that we can then export as a .wav-file
			sound_export.export(
				"{}/{}_{}.wav".format(self.store_directory,export_name,self.enumerate),
				format="wav")
			self.enumerate += 1
			# now we increase the beginning and end of the segment by the user-chosen
			# segment_length in order to get to the next segment
			segment_start += self.segment_length*1000
			segment_end += self.segment_length*1000
		print("Successfully splitted file {} into segments".format(file_name))
