import essentia
import essentia.standard as es
import numpy as np
import librosa
from spafe.features.bfcc import bfcc
from spafe.features.lpc import lpc
import corpus_maker
import pickle
import os
import shutil

def move_file_to_removed(filename):
	default_directory = '/mnt/d/ubc/miles/corpus/'
	removed_directory = os.path.join(default_directory, 'removed')
	
	# Create the 'removed' directory if it doesn't exist
	if not os.path.exists(removed_directory):
		os.makedirs(removed_directory)
	
	source_path = os.path.join(default_directory, filename)
	destination_path = os.path.join(removed_directory, filename)
	
	try:
		# Move the file to the 'removed' directory
		shutil.move(source_path, destination_path)
		print(f"File '{filename}' has been moved to the 'removed' directory.")
	except FileNotFoundError:
		print(f"File '{filename}' does not exist in the default directory.")
	except Exception as e:
		print(f"An error occurred while moving the file: {e}")

# Example usage:
# move_file_to_removed('example.txt')

our_features = ['lowlevel.dissonance.mean', 'lowlevel.dissonance.stdev', 'lowlevel.erbbands.mean', 'lowlevel.erbbands.stdev', 'lowlevel.gfcc.mean', 'lowlevel.gfcc.stdev', 'lowlevel.hfc.mean', 'lowlevel.hfc.stdev', 'lowlevel.loudness_ebu128.integrated', 'lowlevel.loudness_ebu128.loudness_range', 'lowlevel.loudness_ebu128.momentary.mean', 'lowlevel.loudness_ebu128.momentary.stdev', 'lowlevel.loudness_ebu128.short_term.mean', 'lowlevel.loudness_ebu128.short_term.stdev', 'lowlevel.spectral_centroid.mean', 'lowlevel.spectral_centroid.stdev', 'lowlevel.dynamic_complexity', 'lowlevel.spectral_complexity.mean', 'lowlevel.pitch_salience.mean', 'lowlevel.pitch_salience.stdev', 'lowlevel.spectral_complexity.stdev', 'lowlevel.zerocrossingrate.mean','lowlevel.zerocrossingrate.stdev', 'rhythm.danceability', 'tonal.hpcp.mean', 'tonal.hpcp.stdev', 'lowlevel.mfcc.mean', 'lowlevel.mfcc.stdev', 'lowlevel.spectral_rolloff.mean','lowlevel.spectral_rolloff.stdev', 'lowlevel.spectral_flux.mean', 'lowlevel.spectral_flux.stdev']
our_features2 = ['Envelope_mean', 'Envelope_std', 'Derivative_mean', 'Derivative_std', 'BFCC_mean', 'BFCC_std', 'DerivativeSFX', 'AutoCorrelation', 'FrequencyBands_mean', 'FrequencyBands_std', 'LPC_mean', 'LPC_std', 'NoveltyCurve_mean', 'NoveltyCurve_std', 'PercivalEvaluatePulseTrains', 'KeyExtractor', 'Intensity', 'Larm', 'Loudness', 'LoudnessVickers', 'Inharmonicity_mean', 'Inharmonicity_std', 'OddToEvenHarmonicEnergyRatio_mean', 'OddToEvenHarmonicEnergyRatio_std', 'SpectrumCQ_mean', 'SpectrumCQ_std', 'Tristimulus_1_mean', 'Tristimulus_1_std', 'Tristimulus_2_mean', 'Tristimulus_2_std', 'Tristimulus_3_mean', 'Tristimulus_3_std']

def extract_tonal_features(audio_file):
	# Load the audio file
	loader = es.MonoLoader(filename=audio_file, sampleRate=44100)
	audio = loader()
	
	# Frame generator and windowing
	frame_size = 32768  # Correct frame size for SpectrumCQ
	hop_size = 16384
	window = es.Windowing(type='hann')
	spectrum = es.Spectrum()
	cqt = es.SpectrumCQ()
	spectral_peaks = es.SpectralPeaks()

	# Algorithms to compute the features
	inharmonicity_algo = es.Inharmonicity()
	odd_to_even_algo = es.OddToEvenHarmonicEnergyRatio()
	tristimulus_algo = es.Tristimulus()

	# Lists to store the computed feature values
	inharmonicity_values = []
	odd_to_even_values = []
	spectrum_cq_values = []
	tristimulus_1_values = []
	tristimulus_2_values = []
	tristimulus_3_values = []

	# Process each frame
	for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
		windowed_frame = window(frame)
		frame_spectrum = spectrum(windowed_frame)
		frame_spectrum_cq = cqt(windowed_frame)
		
		# Extract spectral peaks
		frequencies, magnitudes = spectral_peaks(windowed_frame)
		
		# Skip frames where no valid peaks are found
		if len(frequencies) == 0 or len(magnitudes) == 0:
			continue
		
		# Compute inharmonicity with sorted peaks
		try:
			inharmonicity_value = inharmonicity_algo(frequencies, magnitudes)
			odd_to_even_value = odd_to_even_algo(frequencies, magnitudes)
			tristimulus_values = tristimulus_algo(frequencies, magnitudes)
		except RuntimeError as e:
			# Skip frames where calculation fails
			continue
		
		inharmonicity_values.append(inharmonicity_value)
		odd_to_even_values.append(odd_to_even_value)
		spectrum_cq_values.append(frame_spectrum_cq)
		tristimulus_1_values.append(tristimulus_values[0])
		tristimulus_2_values.append(tristimulus_values[1])
		tristimulus_3_values.append(tristimulus_values[2])

	# Convert lists to numpy arrays for easier mean and std computation
	inharmonicity_values = np.array(inharmonicity_values)
	odd_to_even_values = np.array(odd_to_even_values)
	spectrum_cq_values = np.array(spectrum_cq_values)
	tristimulus_1_values = np.array(tristimulus_1_values)
	tristimulus_2_values = np.array(tristimulus_2_values)
	tristimulus_3_values = np.array(tristimulus_3_values)

	# Compute mean and std for each feature
	features = {
		'Inharmonicity_mean': np.mean(inharmonicity_values),
		'Inharmonicity_std': np.std(inharmonicity_values),
		'OddToEvenHarmonicEnergyRatio_mean': np.mean(odd_to_even_values),
		'OddToEvenHarmonicEnergyRatio_std': np.std(odd_to_even_values),
		'SpectrumCQ_mean': np.mean(spectrum_cq_values, axis=0),
		'SpectrumCQ_std': np.std(spectrum_cq_values, axis=0),
		'Tristimulus_1_mean': np.mean(tristimulus_1_values),
		'Tristimulus_1_std': np.std(tristimulus_1_values),
		'Tristimulus_2_mean': np.mean(tristimulus_2_values),
		'Tristimulus_2_std': np.std(tristimulus_2_values),
		'Tristimulus_3_mean': np.mean(tristimulus_3_values),
		'Tristimulus_3_std': np.std(tristimulus_3_values)
	}

	return features

# See all feature names in the pool in a sorted order
#print(sorted(features.descriptorNames()))

def extract_features(audio_file):
	# Load the audio file
	loader = es.MonoLoader(filename=audio_file)
	audio = loader()
	signal, sample_rate = librosa.load(audio_file, sr=None, mono=True)

	features = {}

	# Category: Envelope/SFX
	derivative_sfx = es.DerivativeSFX()(audio)
	features['DerivativeSFX'] = derivative_sfx  # mean/std not applicable

	envelope = es.Envelope()(audio)
	features['Envelope_mean'] = np.mean(envelope)
	features['Envelope_std'] = np.std(envelope)

	# Category: Standard
	autocorrelation = es.AutoCorrelation()(audio)
	features['AutoCorrelation'] = autocorrelation  # mean/std not applicable

	derivative = es.Derivative()(audio)
	features['Derivative_mean'] = np.mean(derivative)
	features['Derivative_std'] = np.std(derivative)

	# Category: Spectral
	bfccs = bfcc(sig=signal, fs=sample_rate)
	features['BFCC_mean'] = np.mean(bfccs, axis=0)
	features['BFCC_std'] = np.std(bfccs, axis=0)

	frequency_bands = es.FrequencyBands()(audio)
	features['FrequencyBands_mean'] = np.mean(frequency_bands, axis=0)
	features['FrequencyBands_std'] = np.std(frequency_bands, axis=0)

	y, sr = librosa.load(audio_file, sr=None)
	lpc_coeffs = librosa.lpc(y, order=12)
	features['LPC_mean'] = lpc_coeffs.mean()
	features['LPC_std'] = lpc_coeffs.std()

	# Category: Rhythm
	novelty_curve = es.NoveltyCurve()(list(es.FrameGenerator(audio, frameSize=1024, hopSize=512)))
	features['NoveltyCurve_mean'] = np.mean(novelty_curve)
	features['NoveltyCurve_std'] = np.std(novelty_curve)

	y, sr = librosa.load(audio_file, sr=None)
	onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
	onset_times = librosa.frames_to_time(onset_frames, sr=sr)
	onset_times_essentia = essentia.array(onset_times)
	percival = es.PercivalEvaluatePulseTrains()(essentia.array(y), onset_times_essentia)
	features['PercivalEvaluatePulseTrains'] = percival  # mean/std not applicable

	key_extractor = es.KeyExtractor()(audio)
	features['KeyExtractor'] = key_extractor  # mean/std not applicable

	# Category: Loudness/dynamics
	intensity = es.Intensity()(audio)
	features['Intensity'] = intensity  # mean/std applicable
	features['Intensity_mean'] = np.mean(intensity)
	features['Intensity_std'] = np.std(intensity)

	larm = es.Larm()(audio)
	features['Larm'] = larm  # mean/std applicable
	features['Larm_mean'] = np.mean(larm)
	features['Larm_std'] = np.std(larm)

	loudness = es.Loudness()(audio)
	features['Loudness'] = loudness  # mean/std applicable
	features['Loudness_mean'] = np.mean(loudness)
	features['Loudness_std'] = np.std(loudness)

	loudness_vickers = es.LoudnessVickers()(audio)
	features['LoudnessVickers'] = loudness_vickers  # mean/std applicable
	features['LoudnessVickers_mean'] = np.mean(loudness_vickers)
	features['LoudnessVickers_std'] = np.std(loudness_vickers)

	return features

# Example usage:
#features2 = extract_features(audiofile)
#features2.update(extract_tonal_features(audiofile))

schema = {'id': 'INTEGER PRIMARY KEY'}
schema2 = {
	'lowlevel_dissonance_mean': 'REAL',
	'lowlevel_dissonance_stdev': 'REAL',
	'lowlevel_erbbands_mean': 'BLOB',
	'lowlevel_erbbands_stdev': 'BLOB',
	'lowlevel_gfcc_mean': 'BLOB',
	'lowlevel_gfcc_stdev': 'BLOB',
	'lowlevel_hfc_mean': 'REAL',
	'lowlevel_hfc_stdev': 'REAL',
	'lowlevel_loudness_ebu128_integrated': 'REAL',
	'lowlevel_loudness_ebu128_loudness_range': 'REAL',
	'lowlevel_loudness_ebu128_momentary_mean': 'REAL',
	'lowlevel_loudness_ebu128_momentary_stdev': 'REAL',
	'lowlevel_loudness_ebu128_short_term_mean': 'REAL',
	'lowlevel_loudness_ebu128_short_term_stdev': 'REAL',
	'lowlevel_spectral_centroid_mean': 'REAL',
	'lowlevel_spectral_centroid_stdev': 'REAL',
	'lowlevel_dynamic_complexity': 'REAL',
	'lowlevel_spectral_complexity_mean': 'REAL',
	'lowlevel_pitch_salience_mean': 'REAL',
	'lowlevel_pitch_salience_stdev': 'REAL',
	'lowlevel_spectral_complexity_stdev': 'REAL',
	'lowlevel_zerocrossingrate_mean': 'REAL',
	'lowlevel_zerocrossingrate_stdev': 'REAL',
	'rhythm_danceability': 'REAL',
	'tonal_hpcp_mean': 'BLOB',
	'tonal_hpcp_stdev': 'BLOB',
	'lowlevel_mfcc_mean': 'BLOB',
	'lowlevel_mfcc_stdev': 'BLOB',
	'lowlevel_spectral_rolloff_mean': 'REAL',
	'lowlevel_spectral_rolloff_stdev': 'REAL',
	'lowlevel_spectral_flux_mean': 'REAL',
	'lowlevel_spectral_flux_stdev': 'REAL',
	'Envelope_mean': 'REAL',
	'Envelope_std': 'REAL',
	'Derivative_mean': 'REAL',
	'Derivative_std': 'REAL',
	'BFCC_mean': 'BLOB',
	'BFCC_std': 'BLOB',
	'DerivativeSFX': 'BLOB',
	'AutoCorrelation': 'BLOB',
	'FrequencyBands_mean': 'REAL',
	'FrequencyBands_std': 'REAL',
	'LPC_mean': 'REAL',
	'LPC_std': 'REAL',
	'NoveltyCurve_mean': 'REAL',
	'NoveltyCurve_std': 'REAL',
	'PercivalEvaluatePulseTrains': 'REAL',
	'KeyExtractor': 'BLOB',
	'Intensity': 'INTEGER',
	'Larm': 'REAL',
	'Loudness': 'REAL',
	'LoudnessVickers': 'REAL',
	'Inharmonicity_mean': 'REAL',
	'Inharmonicity_std': 'REAL',
	'OddToEvenHarmonicEnergyRatio_mean': 'REAL',
	'OddToEvenHarmonicEnergyRatio_std': 'REAL',
	'SpectrumCQ_mean': 'BLOB',
	'SpectrumCQ_std': 'BLOB',
	'Tristimulus_1_mean': 'REAL',
	'Tristimulus_1_std': 'REAL',
	'Tristimulus_2_mean': 'REAL',
	'Tristimulus_2_std': 'REAL',
	'Tristimulus_3_mean': 'REAL',
	'Tristimulus_3_std': 'REAL'}
schema3 = {'Intensity_mean': 'REAL', 'Intensity_std': 'REAL', 'Larm_mean': 'REAL', 'Larm_std': 'REAL', 'LoudnessVickers_mean': 'REAL', 'LoudnessVickers_std': 'REAL', 'Loudness_mean': 'REAL', 'Loudness_std': 'REAL'}
schema2.update(schema3)
schema2 = dict(sorted(schema2.items()))
schema.update(schema2)

#corpus_maker.delete_table("test.db", "features")

corpus_maker.create_table("test.db", "features", schema)

# Open the file in read mode
with open('num.txt', 'r') as file:
	# Read the integer value (assuming it's the only content in the file)
	i_start = int(file.read().strip())

for i in range(i_start, 3964):
	try:
		# Define the audio file path
		audiofile = f"/mnt/d/ubc/miles/corpus/{i+1}.wav"

		# Compute all features.
		# Aggregate 'mean' and 'stdev' statistics for all low-level, rhythm, and tonal frame features.
		features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
			rhythmStats=['mean', 'stdev'],
			tonalStats=['mean', 'stdev'],
			gfccStats=['mean','stdev'],
			mfccStats=['mean','stdev'])(audiofile)
		pool_dict = {descriptor: features[descriptor] for descriptor in features.descriptorNames()}
		pool_dict = {key: pool_dict[key] for key in our_features if key in pool_dict}
		pool_dict.update(extract_features(audiofile))
		pool_dict.update(extract_tonal_features(audiofile))
		pool_dict = dict(sorted(pool_dict.items()))

		row_data = (i,)
		for j in pool_dict.keys():
			if isinstance(pool_dict[j], tuple):
				pool_dict[j] = np.array(pool_dict[j])
			if isinstance(pool_dict[j], (np.ndarray, list)):
				row_data = row_data + (pickle.dumps(pool_dict[j]),)
				continue
			row_data = row_data + (pool_dict[j],)

		corpus_maker.insert_values("test.db", "features", [row_data])
		print(f"Done file {i}.wav")

	except RuntimeError as re:
		fn = str(i+1) + ".wav"
		print(f"exception {re} about file {fn}")
		move_file_to_removed(fn)

	with open('num.txt', 'w') as file:
		# Write the integer to the file as a string
		file.write(str(i+1))
