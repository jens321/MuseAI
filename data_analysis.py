from music21 import *
import matplotlib as plt
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
HORIZONTAL_BAR = 'horizontal_bar'

def plot_accuracy():
    numTrain = [10, 50, 100, 500, 1000]
    for num in numTrain:




def visualize_input_output():
    '''
    Get training music and graph all notes for that specified voice
    Get the input (test) music and graph the specified voice and number of notes
    Get the output (predicted) music and graph it
    '''
    start = 10;

    # Training data
    music_gen = MusicGenerator(training_music, test_music)


    training_notes = music_gen.get_music21_notes()
    parsed_training_notes = []
    for sublist in music_gen.get_parsed_notes(training_notes):
        parsed_training_notes.extend(sublist)
    print(parsed_training_notes)
    training_midi = music_gen.save_to_midi(parsed_training_notes[start:])
    plot(training_midi, " Training ", HISTOGRAM)

    # Test Data
    input_notes = music_gen.get_music21_notes(isTraining = 0)   # use test data
    parsed_input_notes = music_gen.get_parsed_notes(input_notes)[0]
    print(parsed_input_notes)
    # Plot input dataset
    test_midi = music_gen.save_to_midi(parsed_input_notes[start:])#corpus.parse(test_music[0])
    plot(test_midi, " Test Midi ", HISTOGRAM)

    # Get output and graph
    predicted = music_gen.generate_music()
    print(predicted)
    output_midi = music_gen.save_to_midi(predicted)
    plot(output_midi, " Predicted Midi ", HISTOGRAM)

def plot(midi, graph_title = "Graph", graph_type = ''):
    '''
    Generic function to plot the midi files
    '''
    final_graph_title = graph_title + graph_type
    if graph_type == HISTOGRAM:
        midi.plot(HISTOGRAM, 'pitch', title = final_graph_title)
    elif graph_type == HORIZONTAL_BAR:
        midi.plot(HORIZONTAL_BAR, title = final_graph_title)       # TODO: Does not work , idk why
    else:
        midi.plot(title = final_graph_title)

def sample_plot():
    chopin = corpus.parse('chopin/mazurka')
    chopin.plot(HISTOGRAM, 'pitch')

def main():
    visualize_input_output()


if __name__ == '__main__':
    main()
