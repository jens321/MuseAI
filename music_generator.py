from sklearn.ensemble import RandomForestClassifier
from music21 import *
import random
import os
from tqdm import tqdm

def get_music_data():
  '''
  Load the Bach corpus and split the data into training and test. 
  '''
  bach_songs = corpus.getComposer('bach')
  song_list = []
  tqdm_iter = tqdm(total = 200, desc="Getting Music Data...", initial=1)
  trained_songs = 1
  idx = 0
  while trained_songs < 200:
    song = None
    if(os.name == 'posix'):
      song = 'bach/' + '.'.join(str(bach_songs[idx]).split('/')[-1].split('.')[:-1])
    else:
      song = 'bach/' + '.'.join(str(bach_songs[idx]).split('\\')[-1].split('.')[:-1])
    # Check if Soprano voice exits
    parsed_song = corpus.parse(song)
    
    # Hack to test if the song has a soprano voice
    try:
      part = parsed_song.parts.stream()['soprano']
      song_list.append(song)
      trained_songs += 1
      tqdm_iter.update(1)
    except:
      pass
    idx += 1
  tqdm_iter.close()

  # Randomize the songs before making training and test split 
  random.shuffle(song_list)
  return (song_list[:160], song_list[160:])

class MusicGenerator():
  def __init__(self, training_music, test_music):
    self.training_music = training_music
    self.test_music = test_music

  def generate_music(self):
    '''
    TODO
    '''
    # Parse training music into notes
    music21_notes_train = self.get_music21_notes(self.training_music)
    parsed_notes_train = self.get_parsed_notes(music21_notes_train)

    # Parse test music into notes 
    music21_notes_test = self.get_music21_notes(self.test_music)
    parsed_notes_test = self.get_parsed_notes(music21_notes_test)

    # Create the vocabulary
    # NOTE: To avoid missing key errors, we add all notes from the 
    #       testing set also in the vocab. 
    self.vocab = set(note for group in parsed_notes_train for note in group)
    for group in parsed_notes_test:
      for note in group:
        self.vocab.add(note)

    # Create note to int and int to note mappings
    self.note_to_idx = {note: idx for idx, note in enumerate(self.vocab)}
    self.idx_to_note = {idx: note for note, idx in self.note_to_idx.items()}

    # Create the dataset with notes X and labels Y
    X, Y = self.make_dataset(parsed_notes_train)
    
    # Traing the classifier
    clf = self.train_rf(X, Y)

    # Pick a random song from the test set which we
    # want to listen to
    show_song = self.test_music[random.randint(0, len(self.test_music))]

    # Predicted on the randomly picked song
    predicted = self.get_predictions([show_song], clf)

    self.save_to_midi(predicted)
    return predicted

  def get_music21_notes(self, songs, voice='soprano'):
    '''
    Takes in a list of songs (e.g. "Bach") and parses each one.
    Currently, we take the first 'part' of the song and return
    all its notes and chords.
    Returns
    -------
    notes_to_parse: list of Music21 Notes and Chords
    '''

    notes_to_parse = []
    for song in tqdm(songs, desc="Parsing Songs..."):
      parsed_song = corpus.parse(song)
      # We probably want to make this more flexible so
      # it can take in the part we want?
      part = parsed_song.parts.stream()[voice]
      notes_to_parse.append([note for note in part.flat.notes])

    return notes_to_parse

  def get_parsed_notes(self, music21_notes):
    '''
    Takes in the notes and chords that are music21
    classes as a 2D list (collection of notes for each song
    in the training data).
    Returns
    -------
    notes: list of Note and Chord representations that are hashable
    '''
    notes = []
    for note_group in tqdm(music21_notes, desc="Extracting Notes and Chords..."):
      notes.append([])
      for sound in note_group:
        if isinstance(sound, note.Note):
          notes[-1].append(str(sound.pitch))
        elif isinstance(sound, chord.Chord):
          notes[-1].append('.'.join(str(n) for n in sound.normalOrder))

    # [Jens]: I don't think we need normalization here, since all
    # of our features are already on the same scale.

    return notes

  def make_dataset(self, parsed_notes, sequence_length=10):
    '''
    Takes in the parsed notes, which is a 2D list of notes
    for all the songs in the training data.
    Returns
    -------
    X: [[sequence_length], [sequence_length], ...] (number of notes - sequence length times)
    Y: [number of notes - sequence length]
    '''
    X = []
    Y = []
    for song in tqdm(parsed_notes, desc="Constructing Dataset..."):
      int_notes = list(map(lambda t: self.note_to_idx[t], song))
      for i in range(len(int_notes) - sequence_length):
        X.append(int_notes[i:i + sequence_length])
        Y.append(int_notes[i + sequence_length])

    return (X, Y)

  def train_rf(self, X, Y, estimators=100):
    '''
    Train a Random Forest classifier on the dataset
    Returns
    -------
    clf: the trained Random Forest classifier
    '''
    clf = RandomForestClassifier(n_estimators=estimators)
    clf.fit(X, Y)

    return clf

  def get_predictions(self, test_music, clf, start_length=10):
    '''
    Starts with the first 'start_length' notes of the test_music
    and predicts from then on. Every predicted note/chord is appended
    and used for the next prediction (sliding window).
    Returns
    -------
    predicted: the newly predicted song (including start sequence)
    '''
    notes = self.get_parsed_notes(self.get_music21_notes(test_music))[0]
    int_notes = list(map(lambda t: self.note_to_idx[t], notes))
    predicted = int_notes[0: start_length]

    for i in range(len(int_notes) - start_length):
      prediction = clf.predict([predicted[i: i + start_length]])[0]
      predicted.append(prediction)

    return list(map(lambda t: self.idx_to_note[t], predicted))

  def save_to_midi(self, predicted):
    '''
    Convert the predicted output into a midi file
    Literal copy of https://github.com/Skuldur/Classical-Piano-Composer/blob/master/lstm.py
    We're probably going to want to adjust this one to have different offsets etc.
    '''
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in predicted:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    try:
      midi_stream.show()
    except:
      pass
    # midi_stream.write('midi', fp='test_output.mid')
    return midi_stream
