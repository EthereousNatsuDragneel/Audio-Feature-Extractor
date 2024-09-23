from segmenter import Segmenter
import os

"""
This program is an example implimentation of the segmenter. 
We pull segment information then create a label file for audacity to visualize the background/foreground assignments.
"""

sample_rate = 22500
audio_folder = ''
out_folder = ''

def main():
    s1 = Segmenter()
    for filename in os.listdir(audio_folder):
        path = audio_folder + '/' + filename
        windowData = s1.segment(path)
        
        label_string = ''
        for item in windowData:
            label_string += str(item['start'])+'\t'+str(item['end'])+'\t'+str(item['type'])+'\n'
    
        # open text file and write label data
        if('.mp3' in filename):
            labelname = filename.replace('.mp3', '_MPclusering.txt')
        elif('.wav' in  filename):
            labelname = filename.replace('.wav', '_MPclusering.txt')
        f = open(out_folder + '/' + labelname, 'w')
        f.write(label_string)
        f.close()

if __name__ == '__main__':
    main()
