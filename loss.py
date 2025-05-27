import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def labels_to_probs(labels, num_classes):
    """
    Convert a list of lists of positive item indices to tensor of probabilities

    Args:
        labels: a list of lists of positive item indices
        num_classes: number of classes
    """

    batch_size = len(labels)
    
    # Generate sample indices and label values using list comprehensions
    sample_indices = torch.tensor([i for i, sample in enumerate(labels) for _ in sample], dtype=torch.long)
    label_values = torch.tensor([l for sample in labels for l in sample], dtype=torch.long)
    
    # Handle edge case where there are no labels
    if len(sample_indices) == 0:
        return torch.zeros(batch_size, num_classes)
    
    # Compute the number of labels per sample
    num_labels = torch.tensor([len(sample) for sample in labels], dtype=torch.float32)
    
    # Calculate the scale for each label: 1/(num_labels+alpha * num_classes)
    scales = 1.0 / (num_labels[sample_indices])
    
    # Compute linear indices to efficiently accumulate probabilities
    linear_indices = sample_indices * num_classes + label_values
    
    # Use bincount to sum scales for each unique index, then reshape
    flat_probs = torch.bincount(linear_indices, weights=scales, minlength=batch_size * num_classes)
    probs = flat_probs.reshape(batch_size, num_classes)
    
    return probs

def labels_to_vector(labels, num_classes):
    """
    Convert a list of lists of positive item indices to n-hot tensors

    Args:
        labels: a list of lists of positive item indices
        num_classes: number of classes
    """

    batch_size = len(labels)
    # Calculate the lengths of each label list
    lengths = list(map(len, labels))
    # Generate batch indices using repeat_interleave
    batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(lengths))
    # Concatenate all class indices into a single tensor
    class_indices = torch.cat([torch.tensor(lbls) for lbls in labels])
    # Ini ialize the result tensor with zeros
    vector = torch.zeros(batch_size, num_classes, dtype=torch.float32)
    # Set the appropriate positions to 1
    vector[batch_indices, class_indices] = 1

    return vector

def gradient_magnitude(outputs, st_labels, mt_labels, num_classes, device, single_target):
    """
    Compute the average gradient magnitude for the sessions

    There are two type of sessions:
        1. single target session: the session with only one positive item
        2. multi-target session: the session with multiple positive items

    """

    # calculate target distribution
    if single_target:
        target_distribution = labels_to_probs(st_labels, num_classes).to(device) # (batch_size, num_classes)
    else:
        target_distribution = labels_to_probs(mt_labels, num_classes).to(device) # (batch_size, num_classes)

    # calculate the predicted distribution
    predicted_distribution = torch.softmax(outputs, dim=1) # (batch_size, num_classes)

    # calculate the gradient magnitude
    gradient_magnitude = torch.sum(torch.abs(predicted_distribution - target_distribution), dim=1) # (batch_size)

    # create a mask for the single target sessions
    mt_labels_len = [len(mt_label) for mt_label in mt_labels]
    st_mask = [mt_labels_len[i] == 1 for i in range(len(mt_labels_len))]
    st_mask = torch.tensor(st_mask, device=device)

    # calculate the avg gradient magnitude for the single target sessions
    avg_gradient_magnitude_st = torch.sum(gradient_magnitude[st_mask]) / torch.sum(st_mask)

    # calculate the avg gradient magnitude for the multi-target sessions
    avg_gradient_magnitude_mt = torch.sum(gradient_magnitude[~st_mask]) / torch.sum(~st_mask)

    return avg_gradient_magnitude_st, avg_gradient_magnitude_mt


class bce_loss_mt(nn.Module):
    """
    Compute the multi-target binary cross-entropy loss

    for the sample with only one label, the loss is identical to the binary cross-entropy loss

    Args:
        outputs: output scores from the model
        labels: a list of lists of positive item indices
        num_classes: number of classes
        device: device to use for computation
    """
    
    def __init__(self, num_classes, device):
        super(bce_loss_mt, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, outputs, labels, sessions):
        batch_size = len(labels)

        # Process all labels in the batch
        labels_tensors = [torch.tensor(lst, dtype=torch.long, device=self.device) for lst in labels]
        
        # Get unique positives and counts for all samples
        unique_pos_list = []
        counts_list = []
        max_pos = 0
        for pos_labels in labels_tensors:
            unique_pos, counts = torch.unique(pos_labels, return_counts=True)
            unique_pos_list.append(unique_pos)
            counts_list.append(counts)
            max_pos = max(max_pos, len(unique_pos))
        
        # Pad unique_pos and counts to create batched tensors
        pad_unique_pos = [F.pad(up, (0, max_pos - len(up)), value=-1) for up in unique_pos_list] # (batch_size, max_pos) list
        pad_counts = [F.pad(c, (0, max_pos - len(c)), value=0) for c in counts_list] # (batch_size, max_pos) list
        
        unique_pos_batch = torch.stack(pad_unique_pos)  # (batch_size, max_pos)
        counts_batch = torch.stack(pad_counts).float()  # (batch_size, max_pos)
        valid_pos_mask = (unique_pos_batch != -1)       # (batch_size, max_pos)

        # Get one negative sample for each positive sample
        # ensure that the negative sample is not in the positive samples and the current session
        neg_samples = []
        for i in range(batch_size):
            neg_indices = torch.randperm(self.num_classes, device=self.device)[:(max_pos + 1)]
            neg_sample = neg_indices[~torch.isin(neg_indices, unique_pos_batch[i]) & ~torch.isin(neg_indices, sessions[i])]
            neg_samples.append(neg_sample[0])

        neg_samples = torch.stack(neg_samples)  # (batch_size)


        # Get scores for positive samples
        # Gather scores for all samples
        pos_scores = torch.gather(
            outputs, 1, 
            unique_pos_batch.clamp(min=0)  # Clamp to handle -1 padding
        ) * valid_pos_mask  # (batch_size, max_pos)

        # Calculate positive entropy
        pos_prob = torch.sigmoid(pos_scores) # (batch_size, max_pos)
        pos_entropy = - torch.log(pos_prob + 1e-10)  # Avoid log(0) # (batch_size, max_pos)
        pos_entropy = pos_entropy * counts_batch / torch.sum(counts_batch, dim=1, keepdim=True)  # (batch_size, max_pos)
        pos_entropy = pos_entropy * valid_pos_mask  # (batch_size, max_pos)
        pos_entropy = torch.sum(pos_entropy) / batch_size

        # Get scores for negative samples        
        neg_scores = outputs[:, neg_samples]  # (batch_size)

        # Calculate negative entropy
        neg_prob = torch.sigmoid(neg_scores)  # (batch_size)
        neg_entropy = - torch.log(1 - neg_prob + 1e-10) # Avoid log(0) # (batch_size)
        neg_entropy = torch.sum(neg_entropy) / batch_size

        # Calculate the loss
        loss = pos_entropy + neg_entropy

        return loss


class ce_loss_mt(nn.Module):
    """
    Compute the multi-target cross-entropy loss

    for the sample with only one label, the loss is identical to the cross-entropy loss

    Args:
        outputs: output scores from the model
        labels: a list of lists of positive item indices
        num_classes: number of classes
        device: device to use for computation
        cl: use curriculum learning
    """
    
    def __init__(self, num_classes, device, cl=False):
        super(ce_loss_mt, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.cl = cl
        self.num_epochs = 120
        # self.percentages = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0]

    def forward(self, outputs, labels, session_len, epoch):
        train_targets = labels_to_probs(labels, self.num_classes).to(self.device) # (batch_size, num_classes)

        if self.cl:
            # Entropy Rank-based curriculum learning
            # calculate the entropy of the target distribution
            target_entropy = -torch.sum(train_targets * torch.log(train_targets + 1e-10), dim=1) # (batch_size)
            # calculate the cross-entropy
            cross_entropy = -torch.sum(train_targets * torch.log_softmax(outputs, dim=1), dim=1) # (batch_size)
            # KL divergence
            kl_divergence = cross_entropy - target_entropy

            # rank the samples by their entropy (low to high)
            sorted_indices = torch.argsort(target_entropy, dim=0, descending=False) # (batch_size)

            # # rank the samples by their KL divergence (high to low)
            # sorted_indices = torch.argsort(kl_divergence, dim=0, descending=True) # (batch_size)

            # # rank the samples by the session length (high to low)
            # session_len_tensor = torch.tensor(session_len, device=self.device, dtype=torch.long)
            # sorted_indices = torch.argsort(session_len_tensor, dim=0, descending=True) # (batch_size)

            # # Rank the samples by their entropy (low to high), if the entropy is equal, then rank by the session length (high to low)
            # session_len_tensor = torch.tensor(session_len, device=self.device, dtype=torch.float32)
            # # Convert tensors to numpy arrays for lexicographical sort
            # target_entropy_np = target_entropy.detach().cpu().numpy()
            # session_len_np = session_len_tensor.detach().cpu().numpy()
            # # Use numpy.lexsort: first key (target_entropy) ascending, then -session_len descending
            # sorted_indices_np = np.lexsort((-session_len_np, target_entropy_np))
            # sorted_indices = torch.tensor(sorted_indices_np, device=self.device)

            # # calculate the number of samples to select
            # # 5 steps
            # num_samples = int(self.percentages[epoch // (self.num_epochs // len(self.percentages))] * len(labels))
            # root-p function
            p = 5
            sample_percent = min(1, ((1 - 0.2 ** p) * epoch / 80 + 0.2 ** p) ** (1/p))
            num_samples = int(sample_percent * len(labels))
            # select the samples with the lowest rank
            selected_indices = sorted_indices[:num_samples]
            
            # # hard selection
            # # create a mask for the selected samples
            # mask = torch.zeros(len(labels), dtype=torch.bool, device=self.device)
            # mask[selected_indices] = True
            # # apply the mask to the outputs and targets
            # outputs = outputs[mask]
            # train_targets = train_targets[mask]

            # soft selection
            # for the selected samples, set the temperature to 1.0
            # for the unselected samples, set the temperature to 2.0
            temperature = torch.ones(len(labels), device=self.device) * 2
            temperature[selected_indices] = 1.0
            # apply the temperature to the outputs
            outputs = outputs / temperature.unsqueeze(1)


            # Entropy based weighted curriculum learning
            # calculate the entropy of the target distribution
            # target_entropy = -torch.sum(train_targets * torch.log(train_targets + 1e-10), dim=1)
            # print(target_entropy.max(), target_entropy.min(), target_entropy.mean())

            # calculate the weights for each sample
            # if epoch < 40:
            #     temperature = 0.1
            #     weights = torch.exp(-target_entropy/temperature)
            # elif epoch < 80:
            #     temperature = 0.1 + (10 - 0.1) * (epoch - 40) / (80 - 40)
            #     weights = torch.exp(-target_entropy/temperature)
            # else:
            # #     weights = torch.ones(len(labels), device=self.device)
            # if epoch < 40:
            #     weights = torch.ones(len(labels), device=self.device)
            # elif epoch < 80:
            #     temperature = 10 + (0.1 - 10) * (epoch - 40) / (80 - 40)
            #     weights = torch.exp(-target_entropy/temperature)
            # else:
            #     temperature = 0.1
            #     weights = torch.exp(-target_entropy/temperature)

            # # apply the weights to the train targets
            # train_targets = train_targets * weights.unsqueeze(1)

        loss = nn.CrossEntropyLoss()
        return loss(outputs, train_targets)


class bpr_max_loss_mt(nn.Module):
    """
        Compute the multi-target Bayesian Personalized Ranking (BPR) max loss (with regularization)

        for the sample with only one label, the loss is identical to the BPR-max loss
            proposed in "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"

        Args:
            outputs: output scores from the model (batch_size x num_classes)
            labels: a list of lists of positive item indices (batch_size x num_pos_samples)
            num_classes: number of classes
            device: device to use for computation
            lambda_reg: regularization parameter
            num_neg_samples: number of negative samples to use for each positive sample, if -1 use all
        """
    
    def __init__(self, num_classes, device, lambda_pos=0.01, lambda_reg=0.1, num_neg_samples=-1):
        super(bpr_max_loss_mt, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lambda_pos = lambda_pos
        self.lambda_reg = lambda_reg
        self.num_neg_samples = num_neg_samples if num_neg_samples != -1 else num_classes
        # loss_weight = torch.ones(2, device=device, requires_grad=True)
        # self.loss_weight = nn.Parameter(loss_weight)

    def forward(self, outputs, labels, epoch):
        batch_size = len(labels)
        
        # Sample negatives once for the entire batch
        neg_indices = torch.randperm(self.num_classes, device=self.device)[:self.num_neg_samples]

        # Process all labels in the batch
        labels_tensors = [torch.tensor(lst, dtype=torch.long, device=self.device) for lst in labels]
        
        # Get unique positives and counts for all samples
        unique_pos_list = []
        counts_list = []
        max_pos = 0
        for pos_labels in labels_tensors:
            unique_pos, counts = torch.unique(pos_labels, return_counts=True)
            unique_pos_list.append(unique_pos)
            counts_list.append(counts)
            max_pos = max(max_pos, len(unique_pos))
        
        # Pad unique_pos and counts to create batched tensors
        pad_unique_pos = [F.pad(up, (0, max_pos - len(up)), value=-1) for up in unique_pos_list] # (batch_size, max_pos) list
        pad_counts = [F.pad(c, (0, max_pos - len(c)), value=0) for c in counts_list] # (batch_size, max_pos) list
        
        unique_pos_batch = torch.stack(pad_unique_pos)  # (batch_size, max_pos)
        counts_batch = torch.stack(pad_counts).float()  # (batch_size, max_pos)
        valid_pos_mask = (unique_pos_batch != -1)       # (batch_size, max_pos)

        # Precompute negative masks for all samples
        batch_neg_mask = torch.stack([~torch.isin(neg_indices, pos) for pos in labels_tensors]) # (batch_size, num_neg_samples)
        valid_neg_mask = batch_neg_mask.any(dim=0)      # Ensure at least one sample has valid negatives

        # Get valid negative indices
        valid_neg_indices = neg_indices[valid_neg_mask] 
        num_valid_neg = valid_neg_indices.size(0)
        if num_valid_neg == 0:
            return torch.tensor(0., device=self.device)
        
        batch_neg_mask = batch_neg_mask[:, valid_neg_mask]  # (batch_size, num_valid_neg)

        # Gather scores for all samples
        pos_scores = torch.gather(
            outputs, 1, 
            unique_pos_batch.clamp(min=0)  # Clamp to handle -1 padding
        ) * valid_pos_mask  # (batch_size, max_pos)
        
        neg_scores = outputs[:, valid_neg_indices]  # (batch_size, num_valid_neg)
        # mask neg_scores with -inf
        neg_scores_masked_inf = torch.where(
            batch_neg_mask,
            neg_scores,
            torch.tensor(float('-inf'), device=neg_scores.device).requires_grad_(False)
        )
        # mask neg_scores with 0
        neg_scores_masked_zero = torch.where(
            batch_neg_mask,
            neg_scores,
            torch.tensor(0., device=neg_scores.device).requires_grad_(False)
        )

        # Compute softmax for negatives
        softmax_neg = F.softmax(neg_scores_masked_inf, dim=1)  # (batch_size, num_valid_neg)

        # Compute pairwise differences (broadcasting)
        diff = pos_scores.unsqueeze(2) - neg_scores_masked_inf.unsqueeze(1)  # (batch_size, max_pos, num_valid_neg)
        sigmoid_diff = torch.sigmoid(diff)

        # Apply masks and counts
        weighted = sigmoid_diff * softmax_neg.unsqueeze(1)  # (batch_size, max_pos, num_valid_neg)
        sum_weighted = torch.sum(weighted, dim=2)  # (batch_size, max_pos)
        
        # Apply valid_pos_mask and counts
        masked_sums = sum_weighted * valid_pos_mask # (batch_size, max_pos)
        masked_counts = counts_batch * valid_pos_mask # (batch_size, max_pos)

        # prob = torch.sum(masked_sums * masked_counts) / torch.sum(masked_counts)

        prob = (masked_sums * masked_counts) / torch.sum(masked_counts, dim=1, keepdim=True) # (batch_size, max_pos)
        prob = torch.log(torch.sum(prob, dim=1) + 1e-10)  # Avoid log(0)
        bpr_max_loss = -torch.sum(prob) / batch_size

        # bpr_max_loss = - torch.log(masked_sums + 1e-10)  # Avoid log(0) # (batch_size, max_pos)
        # bpr_max_loss = (bpr_max_loss * masked_counts) / torch.sum(masked_counts, dim=1, keepdim=True)  # (batch_size, max_pos)
        # bpr_max_loss = torch.sum(bpr_max_loss) / batch_size 

        # Regularization term
        reg = self.lambda_reg * (torch.sum(softmax_neg * neg_scores_masked_zero.pow(2)) /  num_valid_neg) / batch_size

        # Loss among positive samples
        pos_scores_i = pos_scores.unsqueeze(2)                # (batch_size, max_pos, 1)
        pos_scores_j = pos_scores.unsqueeze(1)                # (batch_size, 1, max_pos)
        score_diffs = pos_scores_i - pos_scores_j             # (batch_size, max_pos, max_pos)

        # total_counts = counts_batch.sum(dim=1)
        max_counts = counts_batch.max(dim=1).values           # (batch_size, max_pos)

        counts_i = counts_batch.unsqueeze(2)                  # (batch_size, max_pos, 1)
        counts_j = counts_batch.unsqueeze(1)                  # (batch_size, 1, max_pos)

        counts_diffs = counts_i - counts_j                     # (batch_size, max_pos, max_pos)
        target_probs = counts_diffs / max_counts.unsqueeze(1).unsqueeze(2)  # (batch_size, max_pos, max_pos)
        target_probs = torch.sigmoid(target_probs)  # (batch_size, max_pos, max_pos)
        # old_target_probs = target_probs / 2
        # old_target_probs = old_target_probs + 0.5
        
        # if counts_diffs = 0, target_probs = 0.5
        # if counts_diffs > 0, target_probs = 1
        # if counts_diffs < 0, target_probs = 0
        # target_probs = torch.where(counts_diffs > 0,
        #                               torch.tensor(1., device=self.device), 
        #                               torch.tensor(0., device=self.device))
        # target_probs_tie = torch.where(counts_diffs == 0,
        #                                 torch.tensor(0.5, device=self.device), 
        #                                 torch.tensor(0., device=self.device))
        # target_probs = target_probs + target_probs_tie

        # Create mask for valid pairs (both i and j are valid positions and i != j)
        valid_i = valid_pos_mask.unsqueeze(2)                 # (batch_size, max_pos, 1)
        valid_j = valid_pos_mask.unsqueeze(1)                 # (batch_size, 1, max_pos)
        valid_pairs = valid_i & valid_j                       # (batch_size, max_pos, max_pos)

        # Create diagonal mask to exclude comparing a position with itself
        diag_mask = ~torch.eye(max_pos, dtype=torch.bool, device=self.device).unsqueeze(0)
        valid_pairs = valid_pairs & diag_mask                 # (batch_size, max_pos, max_pos)

        # Compute pairwise loss using sigmoid
        pair_loss = target_probs * torch.log(torch.sigmoid(score_diffs) + 1e-10) + (1 - target_probs) * torch.log(1 - torch.sigmoid(score_diffs) + 1e-10)  # (batch_size, max_pos, max_pos)

        # Apply mask and compute mean across valid pairs
        masked_pair_loss = pair_loss * valid_pairs.float()
        num_valid_pairs = valid_pairs.float().sum()
        if num_valid_pairs > 0:
            pos_loss = - masked_pair_loss.sum() / batch_size * 0.0001# * self.lambda_pos
        else:
            pos_loss = torch.tensor(0., device=self.device)

        # # Loss among positive samples
        # pos_scores_i = pos_scores.unsqueeze(2)                # (batch_size, max_pos, 1)
        # pos_scores_j = pos_scores.unsqueeze(1)                # (batch_size, 1, max_pos)
        # score_diffs = pos_scores_i - pos_scores_j             # (batch_size, max_pos, max_pos)

        # total_counts = counts_batch.sum(dim=1)                # (batch_size)

        # counts_i = counts_batch.unsqueeze(2)                  # (batch_size, max_pos, 1)
        # counts_j = counts_batch.unsqueeze(1)                  # (batch_size, 1, max_pos)

        # counts_diffs = counts_i - counts_j                     # (batch_size, max_pos, max_pos)
        # # positive count differences -> 1, negative count differences -> -1
        # unit_count_diffs = torch.where(counts_diffs > 0, 
        #                                torch.tensor(1., device=self.device), 
        #                                torch.tensor(-1., device=self.device))
        
        # counts_prod = counts_i * counts_j                      # (batch_size, max_pos, max_pos)
        # norm_counts_prod = counts_prod / total_counts.unsqueeze(1).unsqueeze(2)  # (batch_size, max_pos, max_pos)
        # norm_counts_prod = norm_counts_prod / total_counts.unsqueeze(1).unsqueeze(2)  # (batch_size, max_pos, max_pos)

        # # Create mask for valid pairs (both i and j are valid positions and i != j)
        # valid_i = valid_pos_mask.unsqueeze(2)                 # (batch_size, max_pos, 1)
        # valid_j = valid_pos_mask.unsqueeze(1)                 # (batch_size, 1, max_pos)
        # valid_pairs = valid_i & valid_j                       # (batch_size, max_pos, max_pos)

        # # Create diagonal mask to exclude comparing a position with itself
        # diag_mask = ~torch.eye(max_pos, dtype=torch.bool, device=self.device).unsqueeze(0)
        # valid_pairs = valid_pairs & diag_mask                 # (batch_size, max_pos, max_pos)

        # # Compute pairwise loss using sigmoid
        # pair_loss = torch.sigmoid(score_diffs * unit_count_diffs)  # (batch_size, max_pos, max_pos)
        # pair_loss = pair_loss * norm_counts_prod                # (batch_size, max_pos, max_pos)

        # # Apply mask and compute mean across valid pairs
        # masked_pair_loss = pair_loss * valid_pairs.float()
        # num_valid_pairs = valid_pairs.float().sum()
        # if num_valid_pairs > 0:
        #     pos_loss = - torch.log(masked_pair_loss.sum() / batch_size) * self.lambda_pos
        # else:
        #     pos_loss = torch.tensor(0., device=self.device)
        
        # Dynamic loss weighting
        # loss = 0
        # loss += 1 / (2 * self.loss_weight[0] ** 2) * bpr_max_loss + torch.log(1 + self.loss_weight[0] ** 2)
        # loss += 1 / (2 * self.loss_weight[1] ** 2) * pos_loss + torch.log(1 + self.loss_weight[1] ** 2)
        # loss += reg

        # print(bpr_max_loss, pos_loss, reg)

        # print("bpr_max_loss: ", bpr_max_loss.item())
        # print("pos_loss: ", pos_loss.item())
        # print("reg: ", reg.item())

        loss = bpr_max_loss + pos_loss + reg

        return loss
        
class uw_combined_loss(nn.Module):
    """
        Compute the combined loss

        Args:
            outputs: output scores from the model (batch_size x num_classes)
            labels: a list of lists of positive item indices (batch_size x num_pos_samples)
            num_classes: number of classes
            device: device to use for computation
            num_neg_samples: number of negative samples to use for each positive sample, if -1 use all
    """
    
    def __init__(self, num_classes, device, num_neg_samples=-1):
        super(uw_combined_loss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.num_neg_samples = num_neg_samples if num_neg_samples != -1 else num_classes

        loss_weight = torch.ones(2, device=device, requires_grad=True)
        self.loss_weight = nn.Parameter(loss_weight)

        self.ce_loss = ce_loss_mt(num_classes, device)
        self.bpr_max_loss = bpr_max_loss_mt(num_classes, device, num_neg_samples=num_neg_samples)

    def forward(self, outputs, labels):
        ce_loss = self.ce_loss(outputs, labels)
        bpr_max_loss = self.bpr_max_loss(outputs, labels)

        # Dynamic loss weighting
        loss = 0
        loss += 1 / (2 * self.loss_weight[0] ** 2) * ce_loss + torch.log(1 + self.loss_weight[0] ** 2)
        loss += 1 / (2 * self.loss_weight[1] ** 2) * bpr_max_loss + torch.log(1 + self.loss_weight[1] ** 2)

        return loss, ce_loss, bpr_max_loss, self.loss_weight

class ce_loss_mt_autocl(nn.Module):
    """
    Compute the multi-target cross-entropy loss with automatic curriculum learning

    for the sample with only one label, the loss is identical to the cross-entropy loss

    """
    def __init__(self, num_classes, device):
        super(ce_loss_mt_autocl, self).__init__()
        self.num_classes = num_classes
        self.device = device

        # temperature parameters
        # num_samples = 331687
        # temp_tensor = torch.ones(num_samples, device=device)
        # self.temp = nn.Parameter(temp_tensor)

        # self.num_class = 7
        # temp_tensor = torch.ones(self.num_class, device=device)
        # self.temp = nn.Parameter(temp_tensor)

        # self.num_entropy_class = 7
        # entropy_temp_tensor = torch.ones(self.num_entropy_class, device=device)
        # self.entropy_temp = nn.Parameter(entropy_temp_tensor)

        self.num_kl_class = 3
        kl_temp_tensor = torch.ones(self.num_kl_class, device=device)
        self.kl_temp = nn.Parameter(kl_temp_tensor)

        # self.num_len_class = 3
        # len_temp_tensor = torch.ones(self.num_len_class, device=device)
        # self.len_temp = nn.Parameter(len_temp_tensor)

        # parameter for automatic curriculum learning
        # self.linear_1 = nn.Linear(1, 64, bias=True).to(device)
        # self.Tanh_1 = nn.Tanh().to(device)
        # self.linear_2 = nn.Linear(64, 1, bias=True).to(device)
        # self.Tanh2 = nn.Tanh().to(device)

        

    def forward(self, outputs, labels, session_len, epoch):
        # print(list(session_index))
        train_targets = labels_to_probs(labels, self.num_classes).to(self.device) # (batch_size, num_classes)

        # # check gradient direction
        # avg_target_logits = (outputs * train_targets).sum(dim=1) # (batch_size)
        # avg_logits = (outputs * torch.softmax(outputs, dim=1)).sum(dim=1) # (batch_size)
        # diff = avg_target_logits - avg_logits
        # print(f'{diff.mean().item():.4f}, {diff.max().item():.4f}, {diff.min().item():.4f}')

        # calculate the entropy of the target distribution
        target_entropy = -torch.sum(train_targets * torch.log(train_targets + 1e-10), dim=1) # (batch_size)

        # calculate the cross-entropy
        cross_entropy = -torch.sum(train_targets * torch.log_softmax(outputs, dim=1), dim=1) # (batch_size)
        # KL divergence
        kl_divergence = cross_entropy - target_entropy
        # # print(kl_divergence)
        # # print(f"{kl_divergence.mean().item():.4f}, {kl_divergence.max().item():.4f}, {kl_divergence.min().item():.4f}")

        # # # # difficulty measurement
        # # # # difficulty = kl_divergence
        # # # difficulty = target_entropy - kl_divergence * 0.01

        # classify the samples into classes based on their kl divergence
        sorted_kl, sorted_indices = torch.sort(kl_divergence, dim=0, descending=True) # (batch_size)
        # print(sorted_kl)
        num_each_class = len(kl_divergence) // (self.num_kl_class)
        kl_classes = torch.zeros_like(kl_divergence, device=self.device, dtype=torch.long)
        for i in range(self.num_kl_class):
            if i == self.num_kl_class - 1:
                kl_classes[sorted_indices[i*num_each_class : ]] = i
            else:
                kl_classes[sorted_indices[i*num_each_class : (i+1)*num_each_class]] = i
        # get the temperature
        kl_temperature = self.kl_temp[kl_classes]
        print(f'kl_temp: {self.kl_temp}')


        # _, sorted_indices = torch.sort(target_entropy, dim=0, descending=False) # (batch_size)

        # # classify the samples into classes based on their entropy
        # entropy_classes = torch.zeros_like(target_entropy, device=self.device, dtype=torch.long)
        # num_zero_entropy = len(target_entropy[target_entropy < 1e-5])
        # num_each_class = (len(target_entropy) - num_zero_entropy) // (self.num_entropy_class-1)
        # for i in range(self.num_entropy_class-1):
        #     if i == self.num_entropy_class - 2:
        #         entropy_classes[sorted_indices[num_zero_entropy + i*num_each_class : ]] = i+1
        #     else:
        #         entropy_classes[sorted_indices[num_zero_entropy + i*num_each_class : num_zero_entropy + (i+1)*num_each_class]] = i+1

        # # get the temperature
        # entropy_temperature = self.entropy_temp[entropy_classes] # (batch_size)
        # print(f'entropy_temp: {self.entropy_temp}')

        # # comparison experiment: use the length of the session to classify the samples
        # session_len_tensor = torch.tensor(session_len, device=self.device, dtype=torch.long)
        # sorted_len, sorted_indices = torch.sort(session_len_tensor, dim=0, descending=True) # (batch_size)
        # # print(sorted_len)
        # len_classes = torch.zeros_like(session_len_tensor, device=self.device, dtype=torch.long)
        # num_each_class = len(session_len) // (self.num_len_class)
        # for i in range(self.num_len_class):
        #     if i == self.num_len_class - 1:
        #         len_classes[sorted_indices[i*num_each_class : ]] = i
        #     else:
        #         len_classes[sorted_indices[i*num_each_class : (i+1)*num_each_class]] = i
        # # get the temperature
        # len_temperature = self.len_temp[len_classes]
        # print(f'len_temp: {self.len_temp}')

        # # Rank the samples by their entropy and session length
        # target_entropy = -torch.sum(train_targets * torch.log(train_targets + 1e-10), dim=1) # (batch_size)
        # session_len_tensor = torch.tensor(session_len, device=self.device, dtype=torch.float32)
        # # Convert tensors to numpy arrays for lexicographical sort
        # target_entropy_np = target_entropy.detach().cpu().numpy()
        # session_len_np = session_len_tensor.detach().cpu().numpy()

        # # Use numpy.lexsort: first key (target_entropy) ascending, then (-session_len) descending
        # # sorted_indices_np = np.lexsort((-session_len_np, target_entropy_np))
        # # Use numpy.lexsort: first key (-session_len) descending, then (target_entropy) ascending
        # sorted_indices_np = np.lexsort((target_entropy_np, -session_len_np))
        
        # sorted_indices = torch.tensor(sorted_indices_np, device=self.device)

        # entropy_len_classes = torch.zeros_like(target_entropy, device=self.device, dtype=torch.long)
        # num_each_class = len(target_entropy) // (self.num_entropy_class)
        # for i in range(self.num_entropy_class):
        #     if i == self.num_entropy_class - 1:
        #         entropy_len_classes[sorted_indices[i*num_each_class : ]] = i
        #     else:
        #         entropy_len_classes[sorted_indices[i*num_each_class : (i+1)*num_each_class]] = i

        # # get the temperature
        # temperature = self.temp[entropy_len_classes]
        # print(self.temp)
        

        # # calculate the temperature for each sample
        # temperature = self.linear_1(target_entropy.unsqueeze(1)) # (batch_size, 64)
        # temperature = self.Tanh_1(temperature)
        # temperature = self.linear_2(temperature)
        # # temperature = self.Tanh2(temperature) + 1
        # print(f"{temperature.mean().item():.4f}, {temperature.max().item():.4f}, {temperature.min().item():.4f}")
        # # apply the temperature to the logits
        # outputs = outputs / temperature

        # # get the temperature for the current sample
        # temperature = self.temp[list(session_index)] # (batch_size)
        # # print(f"{temperature.mean().item():.4f}, {temperature.max().item():.4f}, {temperature.min().item():.4f}")

        # # compare experiment
        # batch_size = len(labels)
        # random_classes = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        # num_each_class = len(labels) // (self.num_class)
        # for i in range(self.num_class):
        #     if i == self.num_class - 1:
        #         random_classes[i*num_each_class : ] = i
        #     else:
        #         random_classes[i*num_each_class : (i+1)*num_each_class] = i
        # # get the temperature
        # temperature = self.temp[random_classes]
        # print(f"random_temp: {self.temp}")


        # apply the temperature to the logits
        temperature = kl_temperature
        outputs = outputs / temperature.unsqueeze(1)

        # calculate the loss
        loss = nn.CrossEntropyLoss()
        ce_loss = loss(outputs, train_targets)
        # regularization term
        reg = 1e-3 * (torch.log(temperature+1e-10)**2).sum()
        # reg = 0
        # total loss
        total_loss = ce_loss + reg
        return total_loss
    

# define minimizer functions
# input: kl_divergence tensor, shape (batch_size), lambda float
# output: weight tensor, shape (batch_size)
def cauchy_minimizer(kl_divergence, spl_lambda):
    """
    Cauchy minimizer function
    """
    weight = 1 / (1 + torch.pow(kl_divergence / spl_lambda, 2))
    return weight

def welsch_minimizer(kl_divergence, spl_lambda):
    """
    Welsch minimizer function
    """
    weight = torch.exp(-torch.pow(kl_divergence / spl_lambda, 2))
    return weight

def hard_minimizer(kl_divergence, spl_lambda):
    """
    Hard minimizer function
    """
    weight = torch.zeros_like(kl_divergence)
    weight[kl_divergence < spl_lambda] = 1
    return weight

class ce_loss_mt_spl(nn.Module):
    """
    Compute the multi-target cross-entropy loss with self-paced learning
    """
    def __init__(self, num_classes, device):
        super(ce_loss_mt_spl, self).__init__()
        self.num_classes = num_classes
        self.device = device

        self.min_func_type = 'cauchy' # 'none', 'welsch', 'cauchy', 'hard'

        if self.min_func_type == 'none':
            self.spl_lambda = 1
            self.mu = 1.05
        elif self.min_func_type == 'welsch':
            self.spl_lambda = 6
            self.mu = 1.02
        elif self.min_func_type == 'cauchy':
            self.spl_lambda = 5
            self.mu = 1.02
        elif self.min_func_type == 'hard':
            self.spl_lambda = 10
            self.mu = 1.01

        self.epoch = 0

    def forward(self, outputs, labels, epoch):
        train_targets = labels_to_probs(labels, self.num_classes).to(self.device) # (batch_size, num_classes)
        # calculate the entropy of the target distribution
        target_entropy = -torch.sum(train_targets * torch.log(train_targets + 1e-10), dim=1)
        # calculate the cross-entropy
        cross_entropy = -torch.sum(train_targets * torch.log_softmax(outputs, dim=1), dim=1)
        # KL divergence
        kl_divergence = cross_entropy - target_entropy
        # kl_divergence = cross_entropy
        kl_divergence = kl_divergence.detach()
        print(f"kl_divergence: {kl_divergence.mean().item():.4f}, {kl_divergence.max().item():.4f}, {kl_divergence.min().item():.4f}")

        # update the lambda parameter every epoch
        if epoch > self.epoch:
            self.epoch = epoch
            self.spl_lambda = self.spl_lambda * self.mu

        # calculate the weight for each sample
        if self.min_func_type == 'none':
            weight = torch.ones_like(kl_divergence)
        elif self.min_func_type == 'cauchy':
            weight = cauchy_minimizer(kl_divergence, self.spl_lambda)
        elif self.min_func_type == 'welsch':
            weight = welsch_minimizer(kl_divergence, self.spl_lambda)
        elif self.min_func_type == 'hard':
            weight = hard_minimizer(kl_divergence, self.spl_lambda)
        else:
            raise ValueError('Invalid minimizer function type')
        print(f"weight: {weight.mean().item():.4f}, {weight.max().item():.4f}, {weight.min().item():.4f}")

        # apply the weight to the loss
        if self.min_func_type == "hard":
            # hard selection
            mask = torch.zeros_like(weight, device=self.device, dtype=torch.bool)
            mask[weight > 0] = True
            loss = cross_entropy[mask]
        else:
            loss = cross_entropy * weight
        # calculate the mean loss
        loss = torch.mean(loss)

        return loss
    

class focal_loss_mt(nn.Module):
    """
    Compute the multi-target focal loss
    """
    def __init__(self, num_classes, device, gamma=2):
        super(focal_loss_mt, self).__init__()
        self.num_classes = num_classes
        self.device = device

        self.gamma = gamma

    def forward(self, outputs, labels):
        target_probs = labels_to_probs(labels, self.num_classes).to(self.device) # (batch_size, num_classes)
        # non_zero_mask = target_probs > 0
        # non_zero_mask = non_zero_mask.to(self.device)
        # max_abs = torch.max(target_probs, 1-target_probs)
        # max_abs = max_abs.to(self.device)
        probs = torch.softmax(outputs, dim=1) # (batch_size, num_classes)
        
        # # calculate the modulating factor
        # prob_diff = target_probs - probs
        # # scale <- max(target_probs, 1-target_probs)
        # # scale = 1/2 + torch.abs(prob_diff - 1/2)
        # scale = torch.max(target_probs, 1-target_probs)
        # scaled_prob_diff = prob_diff / scale
        # factors = torch.pow(torch.abs(scaled_prob_diff), self.gamma) # (batch_size, num_classes)

        # # calculate the focal loss
        # loss = -torch.sum(target_probs * factors * torch.log_softmax(outputs, dim=1), dim=1) # (batch_size)

        # loss = torch.mean(loss)

        # calculate the cross-entropy
        cross_entropy = -torch.sum(target_probs * torch.log_softmax(outputs, dim=1), dim=1) # (batch_size)

        # calculate the modulating factor
        # l1 distance between target_probs and probs
        l1_dist = torch.sum(torch.abs((target_probs - probs) * target_probs), dim=1) # (batch_size)
        # l1_dist = l1_dist / torch.sum(max_abs * non_zero_mask, dim=1)
        factors = torch.pow(l1_dist * 2, self.gamma) # (batch_size)

        # calculate the focal loss
        loss = factors * cross_entropy
        loss = torch.mean(loss)

        return loss




    
    

    


    
