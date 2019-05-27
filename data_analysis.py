from music21 import *
import matplotlib.pyplot as plt
from music_generator import MusicGenerator
from main import *

from music21 import *


'''
Goal of this module: plot musical score to reveal relationships
    - visualize notes, scores, and measures
'''


HISTOGRAM = 'histogram'
COLOR_GRID = 'colorgrid'
HORIZONTAL_BAR = 'horizontal_bar'

def plot_accuracy():
    numTrain = [10, 50, 100, 200, 300]

    accuracy_train = []
    accuracy_test = []

    for num in numTrain:
        training_music, test_music = get_music_data(num)
        music_gen = MusicGenerator(training_music, test_music)
        # Parse training music into notes
        music21_notes_train = get_music21_notes(training_music)
        parsed_notes_train = get_parsed_notes(music21_notes_train)

        # Parse test music into notes
        music21_notes_test = get_music21_notes(test_music)
        parsed_notes_test = get_parsed_notes(music21_notes_test)

        # Create the vocabulary
        # NOTE: To avoid missing key errors, we add all notes from the
        #       testing set also in the vocab.
        vocab = set(note for group in parsed_notes_train for note in group)
        for group in parsed_notes_test:
            for note in group:
                vocab.add(note)

        # Create note to int and int to note mappings
        note_to_idx = {note: idx for idx, note in enumerate(vocab)}
        idx_to_note = {idx: note for note, idx in note_to_idx.items()}

        # Create the dataset with notes X and labels Y
        X, Y = make_dataset(parsed_notes_train, note_to_idx)

        # Traing the classifier
        clf = train_rf(X, Y)

        # Get the training accuracy
        print("Training Accuracy")
        print("-----------------")
        training_accuracy = get_accuracy(training_music, clf, note_to_idx, idx_to_note)
        print(training_accuracy)

        print()

        # Get the test accuracy
        print("Test Accuracy")
        print("-----------------")
        test_accuracy = get_accuracy(test_music, clf, note_to_idx, idx_to_note)
        print(test_accuracy)

        accuracy_train.append(training_accuracy)
        accuracy_test.append(test_accuracy)

    plt.plot(numTrain, accuracy_train, label = 'Training Accuracy', linewidth=2)
    plt.plot(numTrain, accuracy_test, label = 'Test Accuracy', linewidth=2)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Number of Songs')
    plt.show()

def visualize_input_output():
    '''
    Get training music and graph all notes for that specified voice
    Get the input (test) music and graph the specified voice and number of notes
    Get the output (predicted) music and graph it
    '''
    start = 10
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

    # Training data
    music_gen = MusicGenerator(training_music, test_music)

    training_notes = music_gen.get_music21_notes(music_gen.training_music)
    parsed_training_notes = []
    for sublist in music_gen.get_parsed_notes(training_notes):
        parsed_training_notes.extend(sublist)
    print(parsed_training_notes)
    training_midi = music_gen.save_to_midi(parsed_training_notes[start:])
    plot(training_midi, " Training ", HISTOGRAM)

    # Test Data
    input_notes = music_gen.get_music21_notes(music_gen.test_music)   # use test data
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
    print('begin plot function')
    final_graph_title = graph_title + graph_type
    if graph_type == HISTOGRAM:
        midi.plot(HISTOGRAM, 'pitch', title = final_graph_title)
    elif graph_type == HORIZONTAL_BAR:
        midi.plot(HORIZONTAL_BAR, title = final_graph_title)       # TODO: Does not work , idk why
    else:
        midi.plot(title = final_graph_title)
    print('end plot function')

def sample_plot():
    chopin = corpus.parse('chopin/mazurka')
    chopin.plot(HISTOGRAM, 'pitch')

def main():
    # plot_accuracy()
    visualize_input_output()


if __name__ == '__main__':
    main()
