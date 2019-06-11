from music21 import * 
import random
import statistics as stats
import argparse
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from util import get_music_data, get_music21_notes, get_parsed_notes, play_music

N_HIDDEN = 128
WEIGHT_PATH = "./rnn_weights/weights.pth"
MODEL_PATH = "./rnn_weights/model.pth"

random.seed(1)
torch.manual_seed(1)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def convert_note_to_one_hot(int_note: int, num_classes: int):
  return [1 if i == int_note else 0 for i in range(num_classes)]

def convert_one_hot_to_note(one_hot: list):
  return one_hot.index(1)

def convert_song_to_one_hot(song: [str], num_classes: int, note_to_idx: dict):
  """
  TODO
  """
  one_hot_notes = []
  for note in song:
    one_hot_note = convert_note_to_one_hot(note_to_idx[note], num_classes)
    one_hot_notes.append(one_hot_note)

  assert len(one_hot_notes) == len(song), 'Song lengths do not match!'

  return one_hot_notes

def make_dataset(parsed_notes, note_to_idx, num_classes):
  '''
  Takes in the parsed notes, which is a 2D list of notes
  for all the songs in the training data.

  Returns
  -------
  [
    song 1: [(x: <1st note>, y: <2nd note>), (x: <2nd note>, y: <3rd note>), ...]
    song 2: [(x: <1st note>, y: <2nd note>), (x: <2nd note>, y: <3rd note>), ...]
    song 3: ... 
  ]
  '''

  dataset = []
  for song in tqdm(parsed_notes, desc="Constructing dataset ..."):
    one_hot_notes = convert_song_to_one_hot(song, num_classes, note_to_idx)
    dataset.append([])

    # -1 since we don't have a predicted "END" token
    for i in range(len(one_hot_notes) - 1):
      dataset[-1].append((one_hot_notes[i], one_hot_notes[i + 1]))

    assert len(dataset[-1]) == len(song) - 1, 'Dataset was wrongly constructed!'

  return dataset 

def train(rnn, song):
  """
  Heavily based on https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html 
  """
  # For every training example, we need to reset the hidden state 
  hidden = rnn.initHidden()

  # Useful to train a classification problem with C classes
  # according to PyTorch documentation
  criterion = nn.NLLLoss()

  # Setting this lower helps avoid the exploding gradient problem
  # Up to 0.005 seems to work
  learning_rate = 0.001

  rnn.zero_grad()

  loss = 0
  for i in range(len(song)):
    input = torch.FloatTensor([song[i][0]])
    output, hidden = rnn(input, hidden)
    l = criterion(output, torch.tensor([np.argmax(song[i][1])]))
    loss += l

  # Compute gradients
  loss.backward()

  # Played around with these values 
  # 0.25 was default in some PyTorch example, but
  # that seems overly cautious
  # lots of explanation came from here: 
  # https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/
  nn.utils.clip_grad_norm_(rnn.parameters(), 100)

  for p in rnn.parameters():
    # learning_rate and grad get multiplied here
    p.data.add_(-learning_rate, p.grad.data)

  return output, loss.item() / len(song)

def train_rnn(rnn, training_dataset):
  n_iters = 100000
  plot_every = 1000

  current_loss = 0
  all_losses = []

  for iter in tqdm(range(1, n_iters + 1), desc="Training RNN ..."):
    sample_song = training_dataset[random.randint(0, len(training_dataset) - 1)]
    output, loss = train(rnn, sample_song)
    current_loss += loss 

    if iter % plot_every == 0:
      print('Average loss:', current_loss/plot_every)
      all_losses.append(current_loss/plot_every)
      current_loss = 0

  plt.figure()
  plt.plot(all_losses)
  plt.savefig('./plots/rnn_training.png')

def predict(random_song, rnn, num_classes, idx_to_note):
  """
  Given a random song, get the first note and let the RNN predict the rest.
  """
  with torch.no_grad():

    predicted_song = []
    hidden = rnn.initHidden()
    output = None 
    for i in range(10):
      note = random_song[i]
      predicted_song.append(idx_to_note[note.index(1)])

      cur_note = torch.FloatTensor([random_song[i]])
      output, hidden = rnn(cur_note, hidden)

    # Append the 10th note and setup for rest of predictions
    # print("OUTPUT")
    # print("------")
    # print(output)
    _, topi = output.topk(1)
    # print(topi)
    predicted_song.append(idx_to_note[topi.item()])
    cur_note = torch.FloatTensor([convert_note_to_one_hot(topi.item(), num_classes)])

    for i in range(10, len(random_song) - 1):
      output, hidden = rnn(cur_note, hidden)

      _, topi = output.topk(1)
      # Convert into a note
      note = idx_to_note[topi.item()]
      predicted_song.append(note)

      cur_note = torch.FloatTensor([convert_note_to_one_hot(topi.item(), num_classes)])

    assert len(predicted_song) == len(random_song), 'Predictions were wrongly made!'

    return predicted_song

def get_accuracy(rnn, music, num_classes, idx_to_note, note_to_idx):
  """
  Calculate training/testing accuracy based on "right or wrong" evaluation
  criterion.

  Returns
  -------
  Mean of training/testing accuracy for each song in the training set
  """
  accuracies = []
  for song in music:
    # Get predicted 
    one_hot_song = convert_song_to_one_hot(song, num_classes, note_to_idx)
    predicted = predict(one_hot_song, rnn, num_classes, idx_to_note)

    count = 0
    assert len(predicted) == len(song), 'Something went wrong when predicting'
    for note1, note2 in zip(predicted[10:], song[10:]):
      if note1 == note2:
        count += 1
    accuracies.append(count/len(song[10:]))

  return stats.mean(accuracies)

  # NOTE
  # For hit@3 etc., just included a set of notes instead of just one
  # and check if note of the original is in the set

def main(): 
  parser = argparse.ArgumentParser(description='PyTorch RNN model')
  parser.add_argument('--train', help='train RNN', action='store_true')
  parser.add_argument('--predict', help='do RNN inference', action='store_true')
  args = parser.parse_args()

  training_music, test_music = get_music_data(200)

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

  # This includes classes only seen in the test data, 
  # we still want to be able to represent them as one hot vectors
  num_classes = len(vocab)

  # Create note to int and int to note mappings
  note_to_idx = {note: idx for idx, note in enumerate(vocab)}
  idx_to_note = {idx: note for note, idx in note_to_idx.items()}

  training_dataset = make_dataset(parsed_notes_train, note_to_idx, num_classes)
  test_dataset = make_dataset(parsed_notes_test, note_to_idx, num_classes)

  if args.train:
    rnn = RNN(num_classes, N_HIDDEN, num_classes)
    train_rnn(rnn, training_dataset)
    # Save weights for reloading later 
    torch.save(rnn, MODEL_PATH)
    rnn.eval()

    print("TRAIN NOTES LEN")
    print('---------------')
    print(parsed_notes_train)
    # Get the training accuracy
    print("Training Accuracy")
    print("-----------------")
    training_accuracy = get_accuracy(rnn, parsed_notes_train, num_classes, idx_to_note, note_to_idx)
    print(training_accuracy)

    # Get the test accuracy
    print("Test Accuracy")
    print("-----------------")
    test_accuracy = get_accuracy(rnn, parsed_notes_test, num_classes, idx_to_note, note_to_idx)
    print(test_accuracy)

  elif args.predict:
    rnn = torch.load(MODEL_PATH)
    # rnn.load_state_dict(torch.load(WEIGHT_PATH))
    rnn.eval()

    print("TRAIN NOTES LEN")
    print('---------------')
    # print(parsed_notes_train)
    # Get the training accuracy
    print("Training Accuracy")
    print("-----------------")
    training_accuracy = get_accuracy(rnn, parsed_notes_train, num_classes, idx_to_note, note_to_idx)
    print(training_accuracy)

    # print()

    # # Get the test accuracy
    # print("Test Accuracy")
    # print("-----------------")
    # test_accuracy = get_accuracy(rnn, test_dataset, num_classes, idx_to_note)
    # print(test_accuracy)


    # random_song = training_dataset[random.randint(0, len(training_dataset) - 1)]
    # predicted = predict(random_song, rnn, num_classes, idx_to_note)
    # print("PREDICTED NOTES")
    # print("---------------")
    # print(predicted)

    # play_music(predicted)

  # Current thought: send one note in, backprop, repeat
  # Add end of song token, or maybe not? => probably not, but mention that 
  # this actually simplifies things on our part 

  # For sampling, would we only start with one note?
  # As time goes, the hidden state will carry more and more
  # of past notes information

if __name__ == "__main__":
  main()

# played around with different dataset sizes
