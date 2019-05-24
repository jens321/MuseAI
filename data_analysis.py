from music21 import *

from music_generator import MusicGenerator


'''
Goal of this module: plot musical score to reveal relationships
    - visualize notes, scores, and measures
'''

training_music = ['bach/bwv66.6',
                    'bach/bwv1.6',
                    'bwv438',
                    'bwv44.7',
                    'bwv436',
                    'bwv89.6',
                    'bwv84.5',
                    'bwv83.5']
# Should probably just stay one?
test_music = ['bach/bwv437']

HISTOGRAM = 'histogram'
COLOR_GRID = 'colorgrid'

def visualizeInputOutput():
    '''
    Get the input (test) music and graph the specified voice and number of notes
    Get the output (predicted) music and graphh it
    '''
    music_gen = MusicGenerator(training_music, test_music)

    # Get input dataset and graph for specified voice (if not specified, default to soprano)
    input_notes = music_gen.get_music21_notes(isTraining = 0)   # use test data
    parsed_input_notes = music_gen.get_parsed_notes(input_notes)[0]

    # Plot input dataset
    test_midi = music_gen.save_to_midi(parsed_input_notes[0:len(parsed_input_notes) - 10])#corpus.parse(test_music[0])
    test_midi.plot(HISTOGRAM, 'pitch')

    # Get output aand graph
    predicted = music_gen.generate_music()
    output_midi = music_gen.save_to_midi(predicted)
    output_midi.plot(HISTOGRAM, 'pitch')

def samplePlot():
    chopin = corpus.parse('chopin/mazurka')
    chopin.plot(HISTOGRAM, 'pitch')

def main():
    visualizeInputOutput()


if __name__ == '__main__':
    main()
