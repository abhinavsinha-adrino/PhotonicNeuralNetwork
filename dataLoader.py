import numpy as np
import torch
import torch.utils.data as data_utils
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import import_vowels

def get_loader(
        batch_size_train, batch_size_test, sr,
        vowel_types=['ae', 'ah', 'aw', 'eh', 'ei', 'er', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw'],
        dir='data/vowels/'):
    
    data, label, _ = import_vowels.load_data(vowel_types, dir, sr)

    max_length = max(len(sublist) for sublist in data)
    if max_length%2 !=0:
        max_length = max_length+1

    # Pad each sublist equally from both ends
    padded_data = []
    for sublist in data:
        pad_left = (max_length - len(sublist)) // 2
        pad_right = max_length - len(sublist) - pad_left
        padded_sublist = np.concatenate(([0] * pad_left, sublist, [0] * pad_right), axis=0)
        padded_data.append(padded_sublist)

    padded_data = np.array(padded_data)

    vowel_dict = OrderedDict()
    vowel_list = np.unique(label)

    for idx, vowel in enumerate(vowel_list):
        vowel_dict[vowel] = idx

    label_id = list(map(vowel_dict.get, label))
    label_id = torch.tensor(label_id)
    # randomize
    p = np.random.permutation(len(label_id))
    padded_data = padded_data[p,:]
    label_id = label_id[p]

    full_dataset = data_utils.TensorDataset(torch.tensor(padded_data), label_id)
    #train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.3])

    # need a stratified split to avoid overfitting
    targets = [full_dataset[i][1] for i in range(len(full_dataset))]  # Extract all labels
    train_idx, test_idx = train_test_split(
        range(len(targets)), test_size=0.1, stratify=targets, random_state=1
    )

    train_dataset = data_utils.Subset(full_dataset, train_idx)
    test_dataset = data_utils.Subset(full_dataset, test_idx)


    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader, max_length