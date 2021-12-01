import librosa
import numpy as np
import soundfile as sf
import torch
from boltons.fileutils import iter_find_files
from torch.utils.data import Dataset


def collate_fn_padd(batch):
    """collate_fn_padd
    Padds batch of variable length

    :param batch:
    """
    # get sequence lengths
    spects = [t[0] for t in batch]
    segs = [t[1] for t in batch]
    lengths = [t[2] for t in batch]

    # pad and stack
    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True)
    lengths = torch.LongTensor(lengths)

    return padded_spects, segs, lengths


def mfcc_dist(mfcc):
    """mfcc_dist
    calc 4-dimensional dist features like in HTK

    :param mfcc:
    """
    d = []
    for i in range(2, 9, 2):
        pad = int(i/2)
        d_i = np.concatenate([np.zeros(pad), ((mfcc[:, i:] - mfcc[:, :-i]) ** 2).sum(0) ** 0.5, np.zeros(pad)], axis=0)
        d.append(d_i)
    return np.stack(d)


def segmentation_to_binary_mask(segmentation):
    """
    replicates boundaries to frame-wise labels
    example:
        segmentation - [0, 3, 5]
        returns - [1, 0, 0, 1, 0, 1]

    :param segmentation:
    :param phonemes:
    """
    mask = torch.zeros(segmentation[-1] + 1).long()
    for boundary in segmentation[1:-1]:
        mask[boundary] = 1
    return mask


def extract_features(wav_file):
    wav, sr = sf.read(wav_file)

    # extract mfcc
    spect = librosa.feature.mfcc(wav,
                                     sr=sr,
                                     n_fft=160,
                                     hop_length=160,
                                     n_mels=40,
                                     n_mfcc=13)

    spect = (spect - spect.mean(0)) / spect.std(0)

    delta  = librosa.feature.delta(spect, order=1)
    delta2 = librosa.feature.delta(spect, order=2)
    spect  = np.concatenate([spect, delta, delta2], axis=0)

    dist = mfcc_dist(spect)
    spect  = np.concatenate([spect, dist], axis=0)

    spect = torch.transpose(torch.FloatTensor(spect), 0, 1)
    return spect


def get_onset_offset(segmentations):
    search_start, search_end = float("inf"), 0
    for seg in segmentations:
        start, end = seg[0], seg[-1]
        if start < search_start:
            search_start = start
        if end > search_end:
            search_end = end
    return search_start, search_end


class WavPhnDataset(Dataset):
    def __init__(self, path):
        self.wav_path = path
        self.data = list(iter_find_files(self.wav_path, "*.wav"))
        super(WavPhnDataset, self).__init__()

    @staticmethod
    def get_datasets():
        raise NotImplementedError

    def process_file(self, wav_path):
        phn_path = wav_path.replace("wav", "PHN")

        # load audio
        spect = extract_features(wav_path)

        # load labels -- segmentation and phonemes
        with open(phn_path, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda line: line.split(" "), lines))

            # get segment times
            times   = torch.FloatTensor([0] + list(map(lambda line: int(line[1]), lines)))
            wav_len = times[-1]
            times   = (times / wav_len * (len(spect) - 1)).long()
            boundries = segmentation_to_binary_mask(times)

        return spect, boundries

    def __getitem__(self, idx):

        spect, seg = self.process_file(self.data[idx])

        return spect, seg, spect.shape[0]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = WavPhnDataset("timit/train")
    spect, boundaries, len = dataset[0]
    print(len(spect[0]))
    print(boundaries)
