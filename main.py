from sklearn.ensemble import RandomForestClassifier
from music21 import * 
# Data preprocessing was heavily inspired by: 
# https://github.com/Skuldur/Classical-Piano-Composer/blob/master/lstm.py

# Random Forrest has to be fitted with two arrays:
# X: [n_samples, n_features]
# Y: [n_samples]

# We'll be representing the notes by integers 
# A -> 0, B -> 1, etc. 
# To do this, we'll need a "vocabulary" of all the notes
# and chords that show up in the training set 

# NOTE: We'll have to somehow be able to deal with
# notes/chords we have never seen (might appear in
# the test set) => how?? 

def get_music21_notes(songs):
  notes_to_parse = []
  for song in songs: 
    parsed_song = corpus.parse(song)
    first_part = parsed_song.parts.stream()[0]
    notes_to_parse.append([note for note in first_part.flat.notes])

  return notes_to_parse

def get_parsed_notes(music21_notes):
  notes = []
  for note_group in music21_notes: 
    notes.append([])
    for sound in note_group:
      if isinstance(sound, note.Note):
        notes[-1].append(str(sound.pitch))
      elif isinstance(sound, chord.Chord):
        notes[-1].append('.'.join(str(n) for n in sound.normalOrder))

  return notes 

def make_dataset(parsed_notes, note_to_idx, sequence_length=10):
  X = []
  Y = []
  for song in parsed_notes: 
    int_notes = list(map(lambda t: note_to_idx[t], song))
    for i in range(len(int_notes) - sequence_length):
      X.append(int_notes[i:i + sequence_length])
      Y.append(int_notes[i + sequence_length])

  return (X, Y)

def train_rf(X, Y, estimators=100):
  clf = RandomForestClassifier(n_estimators=estimators)
  clf.fit(X, Y)

  return clf 

def get_predictions(test_music, clf, note_to_idx, start_length=10):
  notes = get_parsed_notes(get_music21_notes(test_music))[0]
  int_notes = list(map(lambda t: note_to_idx[t], notes))
  predicted = int_notes[0: start_length]

  for i in range(len(int_notes) - start_length):
    prediction = clf.predict([predicted[i: i + start_length]])[0]
    predicted.append(prediction)

  return predicted[start_length:]

def main(): 
  training_music = ['bach/bwv66.6']
  test_music = ['bach/bwv66.6']

  music21_notes = get_music21_notes(training_music)
  parsed_notes = get_parsed_notes(music21_notes)

  vocab = set(note for group in parsed_notes for note in group)

  note_to_idx = {note: idx for idx, note in enumerate(vocab)}
  idx_to_note = {idx: note for note, idx in note_to_idx.items()}

  X, Y = make_dataset(parsed_notes, note_to_idx)
  
  clf = train_rf(X, Y)

  predicted = get_predictions(test_music, clf, note_to_idx)

  print(Y)
  print(predicted)
    
if __name__ == "__main__":
  main()