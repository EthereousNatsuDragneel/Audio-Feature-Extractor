# corpus_maker

The corpus_maker contains tools and functions to systematically identify and classify 6 second (time duration adjustable) audio segments from .wav files. It can traverse through a given directory and process all .wav files that exist, and optionally clean up (delete) the processed files.
Caution: the delete functionality should be used carefully, as files deleted by the function cannot be recovered from the recycle bin.

The code also allows for the classified audio segments' data to be written in a local database using sqlite3. The audio segments (.wav, default 6 second) are written to a destination folder. Example usage in sample_corpus_maker.py

# features_extractor

The features_extractor contains tools and functions to extract given features (such as DerivativeSFX, LoudnessEBUR128, etc) of a given audio segment (.wav file). The program then computes mean and standard for the extracted features, after making any adjustments (if required, eg. handling uneven nested arrays, datatype conversations, etc)

Sample code can be found in feature_extractor.py

# supervisor

The supervisor contains a simple bot to launch other files (such as the corpus maker implementation or feature_extractor). Essentia and the BFSegmenter use up a lot of the machine's resources, and the program is often killed with exit code -9 or -11. The supervisor runs the given file (eg. features_extractor.py) repeatedly, and is able to handle it being terminated by the system. It will reclaim the resources used up, and then relaunch the code from where it was killed. This allows the machine to run the extractor or corpus maker code on thousands of datafiles without human supervision or interaction.

# BFSegmenter

The BFSegmenter segments audio files and classifies each segment as background, foreground, or background with foreground. Additionaly, for each segment the affect is predicted on a scale of valence and arousal.

![Pipeline](/images/pipeline.png)

Sound designers and soundscape composers manually segment audio files into building blocks for use in a composition. **Machine learning (ridge regression)** is used to classify segments in an audio file automatically. The model has a **83.0% true positive classification rate**. 

Russel’s model
suggests all emotions are distributed in a circular space (https://psycnet.apa.org/record/1981-25062-001).
High levels of valence correspond to pleasant sounds while
low valence levels correspond to unpleasant sounds. Further, high levels of arousal correspond to exciting sounds while low levels correspond to calming sounds. **Levels of valence and arousal are quantified using machine learning for emotion prediction (random forest regression)**. The emotion prediction models use a subset of extracted features to predict valence and arousal on a scale from -1 to 1 for each segment in an audio file.

![Affect Accuracy](/images/affect_accuracy.png)

Example implimentation of the segmenter in *extract_audacity_labels.py*.

## Segment format

    bf type
    duration
    start
    end
    features
    arousal
    valence
    bf probabilities

## Dependancies
Essentia - an open-source library for tools for audio and music analysis, description and synthesis. https://essentia.upf.edu/ 

Scikit-learn - a free software machine learning library for the Python programming language.

For full requirements, check *requirements.txt*.

## Authors

 - Miles Thorogood
 - Joshua Kranabetter
 - Manjot Singh

## License

This project is licensed under the MIT License - see the *LICENSE* file for details.
