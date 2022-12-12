from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, Sampler
from torchaudio import transforms as T
from torchvision import transforms

PATH_TO_TENSOR_COL = "tensor"


def load_tensor(path):
    features_tensor = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(np.reshape(features_tensor, (-1, 64)))


def pad_or_crop_to_shape(tensor, size, rand_side_pad=True):
    assert len(tensor.shape) == 3
    delta = size - tensor.shape[-1]
    if delta > 0:
        if rand_side_pad:
            start_padding = np.random.randint(delta)
            end_padding = delta - start_padding
            res = nn.functional.pad(tensor, pad=(start_padding, end_padding, 0, 0))
        else:
            res = nn.functional.pad(tensor, pad=(0, delta, 0, 0))

        return res
    else:
        return tensor[..., :size]


def adaptive_padding_collate_fn(batch):
    data = []
    target = []
    max_size = max([tens.shape[-1] for (tens, label) in batch])
    for (tens, label) in batch:
        # crop
        data.append(pad_or_crop_to_shape(tens, max_size, rand_side_pad=True))
        target.append(label)

    return torch.stack(data), torch.tensor(target)


def get_augm_func(time_mask_param=80, freq_mask_param=16, crop_augm_max_cut_size=0):
    """
    Returns function for augmentation in MelEmotionsDataset (augm_transform)
    Returned function's input should have [bs, 1, T] shape

    :param time_mask_param:
    :param freq_mask_param:
    :param crop_augm_max_cut_size: if 0 - random crops are not used
    :return:
    """

    t_masking = T.TimeMasking(time_mask_param=time_mask_param)
    f_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    if crop_augm_max_cut_size != 0:
        # we want random crop with random size,
        # so we should sample crop size for each augm_transform call
        def crop_f(tens):
            crop_delta = np.random.randint(crop_augm_max_cut_size)
            random_crop = transforms.RandomCrop(
                np.array(tens.shape)[1:] - np.array([0, crop_delta])
            )

            return random_crop(tens)

        augm_transform = transforms.Compose([f_masking, t_masking, crop_f])
    else:
        augm_transform = transforms.Compose([f_masking, t_masking])

    return augm_transform


class MelEmotionsDataset(Dataset):
    def __init__(
        self, df, *_, augm_transform=None, get_weights_func=None, base_path=None, **__
    ):
        super().__init__()
        df = df.copy()
        if "label" in df.columns:
            df["label"] = df["label"].apply(int)
        else:
            print('There is no column "label" in the TSV')

        if get_weights_func is None:
            df["sampling_weights"] = 1
        else:
            df["sampling_weights"] = get_weights_func(df)

        # sort by length
        if "wav_length" in df.columns:
            df = df.sort_values("wav_length").reset_index(drop=True)
        else:
            print('There is no column "wav_length" in the TSV')

        self.df = df
        self.augm_transform = augm_transform
        self.feature_col = PATH_TO_TENSOR_COL

        if base_path is not None:
            base_path = Path(base_path)
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.df.iloc[idx][self.feature_col]
        if self.base_path is not None:
            path = self.base_path / path

        tens = torch.from_numpy(np.load(path))
        label = self.df.iloc[idx]["label"]

        if self.augm_transform is not None:
            tens = self.augm_transform(tens)

        return tens, label


class LengthWeightedSampler(Sampler[int]):
    def __init__(
        self,
        df,
        batch_size,
        min_length=1,
        max_length=20.5,
        length_delta=0.3,
        decimals=1,
    ):
        # df should be sorted ascending by wav_length
        # we do it in MelEmotionsDataset
        if "wav_length" not in df.columns:
            raise ValueError('There is no column "wav_length" in the TSV')

        super().__init__(df)
        self.df = df
        self.batch_size = batch_size
        self.num_samples = (len(df) // batch_size) * batch_size

        all_lengths = np.round(df["wav_length"].values, decimals)
        _max = max(all_lengths)
        _min = min(all_lengths)

        if max_length is None or max_length > _max:
            max_length = _max
        if min_length is None or min_length < _min:
            min_length = _min

        self.min_length = min_length
        self.max_length = max_length
        self.length_delta = length_delta

        self.decimals = decimals
        self.length_step = np.round(0.1 ** decimals, decimals)

        # is needed to sample batches with max length inclusive
        max_plus_delta = np.round(self.max_length + self.length_step, decimals)

        length_to_index_mapping = {}
        temp_length = 0

        for i, v in enumerate(all_lengths):
            if v > temp_length:
                if v != temp_length + self.length_step:
                    for j in np.arange(
                        temp_length + self.length_step, v, self.length_step
                    ):
                        length_to_index_mapping[np.round(j, decimals)] = i

                length_to_index_mapping[v] = i

                temp_length = v

        # fix to sample batches with max length inclusive
        length_to_index_mapping[
            np.round(np.max(all_lengths) + self.length_step, decimals)
        ] = len(df)

        self.length_to_index_mapping = length_to_index_mapping

        # starts with MIN_LENGTH
        self.lengths, self.lengths_count = np.unique(
            all_lengths[
                length_to_index_mapping[self.min_length] : length_to_index_mapping[
                    max_plus_delta
                ]
            ],
            return_counts=True,
        )

        self.key_length_sampler = Categorical(
            probs=torch.from_numpy(self.lengths_count)
        )

    def __iter__(self):
        N = 0
        res_indexes = []

        while N < self.num_samples:
            key_length = self.lengths[self.key_length_sampler.sample().item()]

            batch_min_length = np.round(
                max(self.min_length, key_length - self.length_delta), self.decimals
            )
            batch_max_length = np.round(
                min(self.max_length, key_length + self.length_delta), self.decimals
            )
            batch_max_length_plus_delta = np.round(
                batch_max_length + self.length_step, self.decimals
            )

            sub_df = self.df.iloc[
                self.length_to_index_mapping[
                    batch_min_length
                ] : self.length_to_index_mapping[batch_max_length_plus_delta]
            ][["sampling_weights"]]

            sampling_weights = torch.from_numpy(
                sub_df.sampling_weights.values.astype(float)
            )
            sub_iloc_indexes = torch.multinomial(
                sampling_weights, self.batch_size, True
            ).tolist()

            batch_indexes = sub_df.iloc[sub_iloc_indexes].index.tolist()
            res_indexes.extend(batch_indexes)

            N += self.batch_size

        return iter(res_indexes)

    def __len__(self):
        return self.num_samples
