"""
Created on 2025-05-21

@author: Tan Weile

This file is used to retrieve the target items for each session in the training dataset.

For each given session which length is n, the n-th item is the target item.
If there exist sessions that the first n-1 items are the same as the given session, the last item of these sessions are also target items.
We need to retrieve these target items for each session in the training dataset.

In addition, we also apply data augmentation technique to the training/validation/test dataset.
For each session s[1:n], we add s[1:n-1], s[1:n-2], ..., s[1:2] to the dataset.

"""

import torch
import numpy as np
import sys
import tqdm
sys.path.append('..')
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import read_dataset, AugmentedDataset

def collate_fn(samples):
    sessions, labels = zip(*samples)

    return sessions, labels

def build_sess2tar_map(train_loader):
    # build a map from session to target item list
    sess2tar_map = {}
    for batch in tqdm.tqdm(train_loader):
        sessions, labels = batch
        for i in range(len(sessions)):
            # Convert list to tuple to make it hashable for dictionary keys
            session_tuple = tuple(sessions[i])
            # if the session is already in the map, add the label to the target list
            if session_tuple in sess2tar_map:
                sess2tar_map[session_tuple].append(labels[i])
            else:
                sess2tar_map[session_tuple] = [labels[i]]
    return sess2tar_map

def output_processed_dataset(data_loader, loader_type, sess2tar_map, folder):
    with open(f"{folder}/{loader_type}.txt", "w") as f2:
        for batch in tqdm.tqdm(data_loader):
            sessions, labels = batch
            output_lines = []
            for i in range(len(sessions)):
                # output complete session (including the session and the target item)
                output_line = []
                input = sessions[i]
                label = labels[i]
                # concatenate the session and the target item to a list
                complete_session = np.concatenate((input, [label]))
                output_line.append(complete_session)

                if loader_type == "train":
                    # retrieve the target items for the session
                    retrieve_items = sess2tar_map[tuple(sessions[i])]
                    output_line.append(retrieve_items)

                output_lines.append(output_line)

            for output_line in output_lines:
                f2.write('\t'.join([','.join(map(str, x)) for x in output_line]) + '\n')
            
if __name__ == "__main__":
    # dataset
    # dataset_name = "diginetica"
    # dataset_path = "../dataset/diginetica"
    # dataset_name = "yoochoose64"
    # dataset_path = "../dataset/yoochoose64"
    # dataset_name = "yoochoose4"
    # dataset_path = "../dataset/yoochoose4"
    # dataset_name = "gowalla"
    # dataset_path = "../dataset/gowalla"
    # dataset_name = "lastfm"
    # dataset_path = "../dataset/lastfm"
    dataset_name = "retailrocket"
    dataset_path = "../dataset/retailrocket"
    train_sessions, valid_sessions, test_sessions, num_items = read_dataset(Path(dataset_path))
    print(f"dataset name: {dataset_name}, #items: {num_items}")

    train_set = AugmentedDataset(train_sessions, padding=False)
    valid_set = AugmentedDataset(valid_sessions, padding=False)
    test_set = AugmentedDataset(test_sessions, padding=False)

    train_loader = DataLoader(
        train_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # build the map from session to target item list
    print("Building the map from session to target item list...")
    sess2tar_map = build_sess2tar_map(train_loader)

    # output the processed dataset
    print("Outputting the processed dataset...")
    folder = dataset_path + "/retrieve"
    Path(folder).mkdir(parents=True, exist_ok=True)
    output_processed_dataset(train_loader, "train", sess2tar_map, folder)
    output_processed_dataset(valid_loader, "valid", None, folder)
    output_processed_dataset(test_loader, "test", None, folder)