from dataset import *
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import torch
from loss import labels_to_probs
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    dataset_path = "dataset/yoochoose4/"

    dataset_name = dataset_path.split('/')[-2]
    if dataset_name in ['diginetica', 'yoochoose4', 'yoochoose64', 'lastfm', 'gowalla', 'retailrocket']:
        with open(dataset_path + 'num_items.txt', 'r') as f:
            n_items = int(f.readline().strip())
    else:
        raise Exception(f'Unknown Dataset! {dataset_name}')

    # Set device to GPU if available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train, valid, test = load_retrieved_data(dataset_path)
    train_data = RecSysDatasetTrain(train)

    train_loader = DataLoader(train_data, 
                              batch_size=1024, 
                              shuffle=True, 
                              collate_fn=partial(collate_fn_train, max_session_len=50))
    
    # Dictionary to store session statistics
    session_stats = {}
    
    # for each session, collect the session length, the number of target items, and the entropy of the target distribution
    # store the results in a dictionary, with the session id as the key
    max_session_length = 19
    for i, (given_session, mt_labels, given_session_len, session_index) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Move tensors to device
        given_session = given_session.to(device)

        # Calculate entropy of target distribution for multi-term labels
        mt_probs = labels_to_probs(mt_labels, n_items) # [batch_size, n_items]
        mt_entropy = -torch.sum(mt_probs * torch.log(mt_probs + 1e-8), dim=1) # [batch_size]
        
        # Process each session in the batch
        batch_size = given_session.size(0)
        for j in range(batch_size):
            session_id = session_index[j]
            if given_session_len[j] > max_session_length:
                session_len = max_session_length + 1
            else:
                session_len = given_session_len[j]
            
            # Get labels for this session
            session_mt_labels = mt_labels[j]

            # Count target items
            num_mt_target_items = len(session_mt_labels)
            
            # Store statistics for this session
            session_stats[session_id] = {
                'session_length': session_len,
                'num_mt_target_items': num_mt_target_items,
                'mt_entropy': mt_entropy[j].item()
            }
    
    os.makedirs("dataset_stat", exist_ok=True)
    os.makedirs(f'dataset_stat/{dataset_name}', exist_ok=True)

    # Save summary statistics
    with open(f'dataset_stat/{dataset_name}/summary_statistics.txt', 'w') as f:
        f.write(f"Total sessions processed: {len(session_stats)}\n")
        f.write(f"Average session length: {np.mean([stats['session_length'] for stats in session_stats.values()]):.4f}\n")
        num_sessions_with_multiple_target_items = sum(1 for stats in session_stats.values() if stats['num_mt_target_items'] > 1)
        f.write(f"Number of sessions with multiple target items: {num_sessions_with_multiple_target_items}, percentage: {num_sessions_with_multiple_target_items / len(session_stats):.4f}\n")
        f.write(f"Average number of target items: {np.mean([stats['num_mt_target_items'] for stats in session_stats.values()]):.4f}\n")
        f.write(f"Average entropy of target distribution: {np.mean([stats['mt_entropy'] for stats in session_stats.values()]):.4f}\n")
    
    # Plot the distribution of the session length, the number of target items, and the entropy of the target distribution
    session_lengths = [stats['session_length'] for stats in session_stats.values()]
    mt_target_counts = [stats['num_mt_target_items'] for stats in session_stats.values()]
    mt_entropies = [stats['mt_entropy'] for stats in session_stats.values()]
    
    # plt.hist(session_lengths, bins=max(session_lengths)-1)
    # plt.title(f'{dataset_name} - Session Lengths')
    # plt.xlabel('Session Length')
    # plt.ylabel('Count')
    # plt.xticks(np.arange(1, max(session_lengths) + 1, 1))
    # plt.tight_layout()
    # plt.savefig(f'dataset_stat/{dataset_name}/session_lengths.png')
    # plt.close()
    
    plt.hist(mt_target_counts, bins=100, alpha=0.7)
    plt.title(f'{dataset_name} - Number of Target Items')
    plt.xlabel('Number of Target Items')
    plt.ylabel('Count')
    plt.xticks(np.arange(1, max(mt_target_counts) + 1, max(mt_target_counts) // 10))
    plt.tight_layout()
    plt.savefig(f'dataset_stat/{dataset_name}/mt_target_counts.png')
    plt.close()
    
    # plot the relationship between the session length and the number of target items
    plt.scatter(session_lengths, mt_target_counts)
    plt.title(f'{dataset_name} - Session Length vs Number of Target Items')
    plt.xlabel('Session Length')
    plt.ylabel('Number of Target Items')
    tick_positions = list(i for i in range(1, max_session_length + 1)) + [max_session_length + 1]
    tick_labels = [str(i) for i in range(1, max_session_length + 1)] + [r'$\geq$20']
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(np.arange(0, max(mt_target_counts) + 1, max(mt_target_counts) // 10))
    plt.tight_layout()
    plt.savefig(f'dataset_stat/{dataset_name}/session_length_vs_mt_target_counts.png')
    plt.close()
    
    # plot the relationship between the session length and the entropy of the target distribution
    plt.scatter(session_lengths, mt_entropies)
    plt.title(f'{dataset_name} - Session Length vs Entropy of Target Distribution')
    plt.xlabel('Session Length')
    plt.ylabel('Entropy of Target Distribution')
    tick_positions = list(i for i in range(1, max_session_length + 1)) + [max_session_length + 1]
    tick_labels = [str(i) for i in range(1, max_session_length + 1)] + [r'$\geq$20']
    plt.xticks(tick_positions, tick_labels)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'dataset_stat/{dataset_name}/session_length_vs_mt_entropy.png')
    plt.close()

    # plot the relationship between the session length and the average entropy of the target distribution
    # Group sessions by length and calculate average entropy for each length
    length_to_entropies = {}
    for length, entropy in zip(session_lengths, mt_entropies):
        if length not in length_to_entropies:
            length_to_entropies[length] = []
        length_to_entropies[length].append(entropy)
    
    avg_entropies_by_length = {length: np.mean(entropies) for length, entropies in length_to_entropies.items()}
    
    lengths = sorted(avg_entropies_by_length.keys())
    avg_entropies = [avg_entropies_by_length[length] for length in lengths]
    
    plt.plot(lengths, avg_entropies, marker='o')
    plt.title(f'{dataset_name} - Session Length vs Average Entropy of Target Distribution')
    plt.xlabel('Session Length')
    plt.ylabel('Average Entropy of Target Distribution')
    tick_positions = list(i for i in range(1, max_session_length + 1)) + [max_session_length + 1]
    tick_labels = [str(i) for i in range(1, max_session_length + 1)] + [r'$\geq$20']
    plt.xticks(tick_positions, tick_labels)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'dataset_stat/{dataset_name}/session_length_vs_avg_mt_entropy.png')
    plt.close()

    # plot the histogram of the length of the sessions with entropy larger than 0
    session_lengths_with_entropy_larger_than_0 = [length for length, entropy in zip(session_lengths, mt_entropies) if entropy > 0]
    # plt.hist(session_lengths_with_entropy_larger_than_0, bins=max(session_lengths_with_entropy_larger_than_0)-1)
    # plt.title(f'{dataset_name} - Histogram of Session Lengths (Entropy > 0)')
    # plt.xlabel('Session Length')
    # plt.ylabel('Frequency')
    # plt.xticks(np.arange(1, max(session_lengths_with_entropy_larger_than_0) + 1, 1))
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'dataset_stat/{dataset_name}/mt_session_lengths_histogram_entropy_larger_than_0.png')
    # plt.close()

    # two histograms in one figure, one for the sessions with entropy larger than 0, one for all sessions
    bins = np.arange(1, max_session_length + 3)
    plt.hist(session_lengths, bins=bins, label='single target', alpha=0.7)
    plt.hist(session_lengths_with_entropy_larger_than_0, bins=bins, label='multi-target', alpha=0.7)
    plt.title(f'{dataset_name} - Histogram of Session Lengths')
    plt.xlabel('Session Length')
    plt.ylabel('Count')
    tick_positions = list(i+0.5 for i in range(1, max_session_length + 1)) + [max_session_length + 1.5]
    tick_labels = [str(i) for i in range(1, max_session_length + 1)] + [r'$\geq$20']
    plt.xticks(tick_positions, tick_labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'dataset_stat/{dataset_name}/session_lengths_histogram.png')
    plt.close()