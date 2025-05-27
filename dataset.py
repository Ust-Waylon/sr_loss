import pickle
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

def read_train(file_path):
    session_labels = pd.read_csv(file_path, sep='\t', header=None)
    given_sessions = session_labels[0].apply(lambda x: list(map(int, x.split(',')))).values
    labels = session_labels[1].apply(lambda x: list(map(int, x.split(',')))).values

    return given_sessions, labels


def read_valid(file_path):
    # each line in the file is a session
    with open(file_path, 'r') as f:
        sessions = f.readlines()
    sessions = [list(map(int, sess.strip().split(','))) for sess in sessions]

    return sessions
    

def load_retrieved_data(folder_path):
    retrieve_folder_name = "retrieve"
    # check if the folder exists
    if not (Path(folder_path) / retrieve_folder_name).exists():
        raise FileNotFoundError(f"Folder {retrieve_folder_name} does not exist in {folder_path}")
    
    train_data_path = Path(folder_path) / retrieve_folder_name / "train.txt"
    valid_data_path = Path(folder_path) / retrieve_folder_name / "valid.txt"
    if not valid_data_path.exists():
        valid_data_path = Path(folder_path) / retrieve_folder_name / "test.txt"
    test_data_path = Path(folder_path) / retrieve_folder_name / "test.txt"

    train_given_sessions, train_labels = read_train(train_data_path)
    train = (train_given_sessions, train_labels)
    valid = read_valid(valid_data_path)
    test = read_valid(test_data_path)

    return train, valid, test

class RecSysDatasetTrain(Dataset):
    def __init__(self, data):
        self.data = data
        print('-' * 50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-' * 50)

    def __getitem__(self, index):
        given_session = self.data[0][index]
        labels = self.data[1][index]
        return given_session, labels, index

    def __len__(self):
        return len(self.data[0])
    
def collate_fn_train(data, max_session_len, single_target=False):
    given_session, labels, index = zip(*data)
    
    if single_target:
        # get the last item in the sessions as the label
        given_session_label = [[sess[-1]] for sess in given_session]
    # given_session_without_label = [sess[:-1] for sess in given_session]
    given_session_without_label = []
    for i, sess in enumerate(given_session):
        if len(sess) > max_session_len + 1:
            given_session_without_label.append(sess[-max_session_len-1:-1])
        else:
            given_session_without_label.append(sess[:-1])

    # get the length of each session
    given_session_len = [len(sess) for sess in given_session_without_label]   

    # pad the sessions to the same length
    padded_given_session = torch.zeros(len(given_session_without_label), max_session_len).long()
    for i, sess in enumerate(given_session_without_label):
        padded_given_session[i, :len(sess)] = torch.LongTensor(sess)

    if single_target:
        return padded_given_session, given_session_label, given_session_len, index
    else:
        return padded_given_session, labels, given_session_len, index
    

def collate_fn_train_st_mt(data, max_session_len):
    given_session, mt_labels, index = zip(*data)
    
    # get the last item in the sessions as the st label
    st_labels = [[sess[-1]] for sess in given_session]
    # given_session_without_label = [sess[:-1] for sess in given_session]
    given_session_without_label = []
    for i, sess in enumerate(given_session):
        if len(sess) > max_session_len + 1:
            given_session_without_label.append(sess[-max_session_len-1:-1])
        else:
            given_session_without_label.append(sess[:-1])

    # get the length of each session
    given_session_len = [len(sess) for sess in given_session_without_label]   

    # pad the sessions to the same length
    padded_given_session = torch.zeros(len(given_session_without_label), max_session_len).long()
    for i, sess in enumerate(given_session_without_label):
        padded_given_session[i, :len(sess)] = torch.LongTensor(sess)

    return padded_given_session, st_labels, mt_labels, given_session_len, index

class RecSysDatasetValid(Dataset):
    def __init__(self, data):
        self.data = data
        print('-' * 50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data)))
        print('-' * 50)

    def __getitem__(self, index):
        given_session = self.data[index]
        return given_session

    def __len__(self):
        return len(self.data)

def collate_fn_valid(data, max_session_len):
    given_session = data

    # get the last item in the sessions as the label
    given_session_label = [sess[-1] for sess in given_session]
    given_session_label = torch.tensor(given_session_label).long()
    # given_session_without_label = [sess[:-1] for sess in given_session]
    given_session_without_label = []
    for sess in given_session:
        if len(sess) > max_session_len:
            given_session_without_label.append(sess[-max_session_len-1:-1])
        else:
            given_session_without_label.append(sess[:-1])

    # get the length of each session
    given_session_len = [len(sess) for sess in given_session_without_label]     

    # pad the sessions to the same length
    padded_given_session = torch.zeros(len(given_session_without_label), max_session_len).long()
    for i, sess in enumerate(given_session_without_label):
        padded_given_session[i, :len(sess)] = torch.LongTensor(sess)

    return padded_given_session, given_session_label, given_session_len