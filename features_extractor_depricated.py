import essentia
import essentia.standard as es
import warnings
import numpy as np
import json
from corpus_maker import create_table, insert_values

def extract_features(features):
	"""
	the function takes a dictionary of features and their corresponding functions
	the function returns a dictionary of features and their values
	"""
	ex_features = {}

	# Extract features
	for feature_name, feature_func in features.items():
		try:
			# Handle features requiring specific input arguments
			if feature_name in ['Danceability']:
				feature_value = feature_func()(audio)[0]
			elif feature_name == 'NoveltyCurve':
				frames = es.FrameGenerator(audio, frameSize=1024, hopSize=512)
				feature_value = feature_func()(list(frames))
			elif feature_name in ['PercivalEvaluatePulseTrains', 'Dissonance', 'HPCP', 'Inharmonicity', 'Tristimulus']:
				# These features require additional arguments: frames and their spectra
				frames = list(es.FrameGenerator(audio, frameSize=1024, hopSize=512))
				feature_value = []
				for frame in frames:
					spec = spectrum(window(frame))
					freqs, mags = spectral_peaks(spec)
					if len(freqs) == len(mags) and len(freqs) > 0:
						if feature_name == 'HPCP':
							feature_value.append(feature_func()(essentia.array(freqs), essentia.array(mags)))
						elif feature_name == 'Inharmonicity':
							if freqs[0] > 0:  # Ensure fundamental frequency is above 0 Hz
								feature_value.append(feature_func()(essentia.array(freqs), essentia.array(mags)))
						elif feature_name == 'Tristimulus':
							feature_value.append(feature_func()(essentia.array(freqs), essentia.array(mags)))
				if feature_value:
					ex_features[feature_name] = feature_value
			elif feature_name == 'SpectrumCQ':
				# SpectrumCQ requires a specific frame size
				frames = es.FrameGenerator(audio, frameSize=32768, hopSize=512)
				feature_value = [feature_func()(frame) for frame in frames]
			elif feature_name == 'Intensity':
				# Intensity should be a single value, not an array
				feature_value = feature_func()(audio)
			elif feature_name == 'LoudnessEBUR128':
				# LoudnessEBUR128 requires audio as stereo samples
				feature_value = feature_func()(stereo_audio)
			else:
				feature_value = feature_func()(audio)
			ex_features[feature_name] = feature_value
			#print(f"{feature_name} - Success, {type(feature_value)}")
		except Exception as e:
			print(f"Error extracting {feature_name}: {e}")
			#if 'feature_value' in locals():
				#print(type(feature_value))

	return ex_features

def mean_sd(data):
	"""
	The function returns a dictionary, where each element
	is named after a feature, and value holds an array.
	The first value in the array is the mean, and second is the std.
	The function handles the datatypes of the features passed in the data.
	The parameter data is passed as a dictionary.
	"""

	mean_and_std = {}

	for feature_name in data:
		feature_value = data[feature_name]
		try:
			if isinstance(feature_value, list) or isinstance(feature_value, tuple):
				feature_value = np.asarray(feature_value)
			elif isinstance(feature_value, float) or isinstance(feature_value, int):
				feature_value = np.array([feature_value])
			elif isinstance(feature_value, np.ndarray):
				pass
			else:
				raise ValueError(f"Unsupported data type for {feature_name}: {type(feature_value)}")

			if feature_name == 'LoudnessEBUR128':
				# Ensure feature_value is a 2D array where each row is a frame and each column is a coefficient
				feature_value = [f for f in feature_value if isinstance(f, np.ndarray) and len(f.shape) == 1 and len(f) > 0]
				if feature_value:
					max_length = max(len(f) for f in feature_value)
					feature_value = [np.pad(f, (0, max_length - len(f)), mode='constant') for f in feature_value]
					feature_value = np.vstack(feature_value)
				else:
					feature_value = np.array(feature_value)
				mean_value = np.mean(feature_value, axis=0)
				std_value = np.std(feature_value, axis=0)

			elif feature_name in ['BFCC', 'GFCC', 'MFCC', 'LPC']:
				# Ensure feature_value is a 2D array where each row is a frame and each column is a coefficient
				feature_value = [f for f in feature_value if len(f.shape) == 1 and len(f) > 0]
				if feature_value:
					max_length = max(len(f) for f in feature_value)
					feature_value = [np.pad(f, (0, max_length - len(f)), mode='constant') for f in feature_value]
					feature_value = np.vstack(feature_value)
				else:
					feature_value = np.array(feature_value)
				mean_value = np.mean(feature_value, axis=0)
				std_value = np.std(feature_value, axis=0)
			else:
				if feature_value.ndim > 1:
					mean_value = np.mean(feature_value, axis=0)
					std_value = np.std(feature_value, axis=0)
				else:
					mean_value = np.mean(feature_value)
					std_value = np.std(feature_value)

			mean_and_std[feature_name] = [mean_value, std_value]
			#print(f"{feature_name} mean and std computed")
		except Exception as e:
			print(f"exception for {feature_name},: {e}")

	return mean_and_std

# Suppress specific runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# List of features
features = {
	'DerivativeSFX': es.DerivativeSFX,
	'Envelope': es.Envelope,
	'AutoCorrelation': es.AutoCorrelation,
	'Derivative': es.Derivative,
	'ZeroCrossingRate': es.ZeroCrossingRate,
	'BFCC': es.BFCC,
	'ERBBands': es.ERBBands,
	'Flux': es.Flux,
	'FrequencyBands': es.FrequencyBands,
	'GFCC': es.GFCC,
	'HFC': es.HFC,
	'LPC': es.LPC,
	'MFCC': es.MFCC,
	'RollOff': es.RollOff,
	'SpectralCentroidTime': es.SpectralCentroidTime,
	'SpectralComplexity': es.SpectralComplexity,
	'Danceability': es.Danceability,
	'NoveltyCurve': es.NoveltyCurve,
	'PercivalEvaluatePulseTrains': es.PercivalEvaluatePulseTrains,
	'Dissonance': es.Dissonance,
	'HPCP': es.HPCP,
	'Inharmonicity': es.Inharmonicity,
	'PitchSalience': es.PitchSalience,
	'SpectrumCQ': es.SpectrumCQ,
	'Tristimulus': es.Tristimulus,
	'DynamicComplexity': es.DynamicComplexity,
	'Intensity': es.Intensity,
	'Larm': es.Larm,
	'Loudness': es.Loudness,
	'LoudnessEBUR128': es.LoudnessEBUR128,
	'LoudnessVickers': es.LoudnessVickers
}

schema = {
	'id': 'INTEGER PRIMARY KEY',
	'DerivativeSFXMean': 'REAL',
	'DerivativeSFXSD': 'REAL',
	'EnvelopeMean': 'REAL',
	'EnvelopeSD': 'REAL',
	'AutoCorrelationMean': 'REAL',
	'AutoCorrelationSD': 'REAL',
	'DerivativeMean': 'REAL',
	'DerivativeSD': 'REAL',
	'ZeroCrossingRateMean': 'REAL',
	'ZeroCrossingRateSD': 'REAL',
	'BFCCMean': 'TEXT',
	'BFCCSD': 'TEXT',
	'ERBBandsMean': 'REAL',
	'ERBBandsSD': 'REAL',
	'FluxMean': 'REAL',
	'FluxSD': 'REAL',
	'FrequencyBandsMean': 'REAL',
	'FrequencyBandsSD': 'REAL',
	'GFCCMean': 'TEXT',
	'GFCCSD': 'TEXT',
	'HFCMean': 'REAL',
	'HFCSD': 'REAL',
	'LPCMean': 'TEXT',
	'LPCSD': 'TEXT',
	'MFCCMean': 'TEXT',
	'MFCCSD': 'TEXT',
	'RollOffMean': 'REAL',
	'RollOffSD': 'REAL',
	'SpectralCentroidTimeMean': 'REAL',
	'SpectralCentroidTimeSD': 'REAL',
	'SpectralComplexityMean': 'REAL',
	'SpectralComplexitySD': 'REAL',
	'DanceabilityMean': 'REAL',
	'DanceabilitySD': 'REAL',
	'NoveltyCurveMean': 'REAL',
	'NoveltyCurveSD': 'REAL',
	'PercivalEvaluatePulseTrainsMean': 'REAL',
	'PercivalEvaluatePulseTrainsSD': 'REAL',
	'DissonanceMean': 'REAL',
	'DissonanceSD': 'REAL',
	'HPCPMean': 'TEXT',
	'HPCPSD': 'TEXT',
	'InharmonicityMean': 'REAL',
	'InharmonicitySD': 'REAL',
	'PitchSalienceMean': 'REAL',
	'PitchSalienceSD': 'REAL',
	'SpectrumCQMean': 'TEXT',
	'SpectrumCQSD': 'TEXT',
	'TristimulusMean': 'TEXT',
	'TristimulusSD': 'TEXT',
	'DynamicComplexityMean': 'REAL',
	'DynamicComplexitySD': 'REAL',
	'IntensityMean': 'REAL',
	'IntensitySD': 'REAL',
	'LarmMean': 'REAL',
	'LarmSD': 'REAL',
	'LoudnessMean': 'REAL',
	'LoudnessSD': 'REAL',
	'LoudnessEBUR128Mean': 'TEXT',
	'LoudnessEBUR128SD': 'TEXT',
	'LoudnessVickersMean': 'REAL',
	'LoudnessVickersSD': 'REAL'
}

#sample usage of the methods:
#ex_features = extract_features(features)
#mstd = mean_sd(ex_features)

create_table("test.db", "features", schema)

# Open the file in read mode
with open('num.txt', 'r') as file:
	# Read the integer value (assuming it's the only content in the file)
	i_start = int(file.read().strip())

for i in range(i_start, 3964):
	# Load the audio file
	filename_string = "/mnt/d/ubc/miles/corpus/" + str(i) + ".wav"
	audio = es.MonoLoader(filename=filename_string)()

	# Convert mono to stereo by duplicating the channel
	stereo_audio = np.vstack((audio, audio)).T

	# Spectrum calculation
	spectrum = es.Spectrum()
	window = es.Windowing(type='hann')
	spectral_peaks = es.SpectralPeaks()

	mstd = mean_sd(extract_features(features))
	row_data = (i,)
	for key in mstd:
		if isinstance(mstd[key][0], np.ndarray):
			#serialize the aray to write them in sql database
			mstd[key][0] = json.dumps(mstd[key][0].tolist())
			mstd[key][1] = json.dumps(mstd[key][1].tolist())
		row_data = row_data + (mstd[key][0], mstd[key][1])

	insert_values("test.db", "features", [row_data])
	print(f"Done file {i}.wav")
	with open('num.txt', 'w') as file:
		# Write the integer to the file as a string
		file.write(str(i+1))
