import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        for tok in ['<pad>', '<unk>', '<bos>', '<eos>']:
            self.add_word(tok)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def build_common_dictionary(path, files, add_bos=False, add_eos=True):
    dic = Dictionary()
    for fname in files:
        fpath = os.path.join(path, fname)
        assert os.path.exists(fpath), f"Missing file: {fpath}"
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split()
                if add_bos:
                    words = ['<bos>'] + words
                if add_eos:
                    words = words + ['<eos>']
                for w in words:
                    dic.add_word(w)
    return dic


def tokenize_with_dictionary(path, filename, dictionary, add_bos=False, add_eos=True):
    ids = []
    fpath = os.path.join(path, filename)
    assert os.path.exists(fpath), f"Missing file: {fpath}"
    unk_id = dictionary.word2idx['<unk>']
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split()
            if add_bos:
                words = ['<bos>'] + words
            if add_eos:
                words = words + ['<eos>']
            for w in words:
                ids.append(dictionary.word2idx.get(w, unk_id))
    return torch.tensor(ids, dtype=torch.long)


class CorpusCommon:
    """
    Builds a common dictionary from train/valid/test and tokenizes all splits with it.
    """
    def __init__(self, path, add_bos=False, add_eos=True):
        self.path = path
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.files = ['train.txt', 'valid.txt', 'test.txt']

        self.dictionary = build_common_dictionary(
            path=self.path, files=self.files, add_bos=self.add_bos, add_eos=self.add_eos
        )

        self.train = tokenize_with_dictionary(self.path, 'train.txt', self.dictionary, self.add_bos, self.add_eos)
        self.valid = tokenize_with_dictionary(self.path, 'valid.txt', self.dictionary, self.add_bos, self.add_eos)
        self.test  = tokenize_with_dictionary(self.path, 'test.txt',  self.dictionary, self.add_bos, self.add_eos)


def save_vocabulary_dict(dictionary, vocab_filename):
    if not vocab_filename.lower().endswith('.txt'):
        vocab_filename += '.txt'
    with open(vocab_filename, 'w', encoding='utf-8') as f:
        for word in dictionary.idx2word:
            f.write(f"{word}\n")
    print(f"Text vocabulary saved to: {vocab_filename}")
    print(f"Vocabulary size: {len(dictionary)}")


# =========================
# DataLoader (batch-first, sliding windows)
# =========================

class GPTDataLoader:
    """
    Batch-first data loader that turns a 1D token tensor into (batch_size, tokens_per_seq),
    then yields overlapping sliding windows of length seq_len as (inputs, targets).
    """
    def __init__(self, corpus_like, batch_size, seq_len):
        self.data = self._prepare_data(corpus_like.data, batch_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = len(corpus_like.dictionary)
        self.pad_token_id = corpus_like.dictionary.word2idx['<pad>']

    @classmethod
    def from_tensor(cls, data_tensor, dictionary, batch_size, seq_len):
        dummy = type('Dummy', (), {})()
        dummy.data = data_tensor
        dummy.dictionary = dictionary
        return cls(dummy, batch_size, seq_len)

    def _prepare_data(self, data, batch_size):
        tokens_per_seq = data.size(0) // batch_size
        total_tokens = tokens_per_seq * batch_size
        data = data.narrow(0, 0, total_tokens)
        data = data.view(batch_size, tokens_per_seq).contiguous()
        return data

    def get_batch(self, start_idx):
        max_seq_len = self.data.size(1)
        end_idx = start_idx + self.seq_len + 1
        if end_idx > max_seq_len:
            raise IndexError(f"start_idx {start_idx} exceeds available windows for seq_len={self.seq_len}")
        chunk = self.data[:, start_idx:end_idx]            # (B, seq_len+1)
        inputs = chunk[:, :-1]                             # (B, seq_len)
        targets = chunk[:, 1:]                             # (B, seq_len)
        return inputs, targets

    def __len__(self):
        return max(0, self.data.size(1) - self.seq_len)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_batch(i)

    @property
    def num_batches(self):
        return len(self)

    def get_stats(self):
        return {
            'total_sequence_length': self.data.size(1),
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'num_batches': self.num_batches,
            'tokens_per_batch': self.batch_size * self.seq_len,
            'total_tokens': self.data.numel(),
            'data_shape': tuple(self.data.shape),
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
            'memory_usage_mb': self.data.numel() * self.data.element_size() / (1024 * 1024)
        }


def get_linear_warmup_decay_lambda_with_floor(total_steps, warmup_steps, min_lr, base_lr):
    """
    Linear warmup -> linear decay with an LR floor (min_lr).
    Returns a lambda for torch.optim.lr_scheduler.LambdaLR.
    """
    floor = max(0.0, min_lr / max(1e-12, base_lr))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(floor, 1.0 - progress)
    return lr_lambda


def get_cosine_warmup_decay_lambda_with_floor(total_steps, warmup_steps, cycles, min_lr, base_lr):
    """
    Cosine warmup -> cosine decay with an LR floor (min_lr).
    Returns a lambda for torch.optim.lr_scheduler.LambdaLR.
    """
    floor = max(0.0, min_lr / max(1e-12, base_lr))
    def lr_lambda(step):
        import math as m
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        value = 0.5 * (1.0 + m.cos(m.pi * 2.0 * cycles * progress))
        return max(floor, value)
    return lr_lambda


# =========================
# Evaluation
# =========================

def evaluate_on_fixed_subset(model, val_loader, device, fixed_indices):
    """
    Evaluate token-true loss/perplexity on a fixed subset of validation window indices.
    Ensures comparability across mid-epoch evaluations.
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    pad_id = getattr(val_loader, "pad_token_id", None)

    with torch.no_grad():
        max_idx = len(val_loader) - 1
        for idx in fixed_indices:
            if not (0 <= idx <= max_idx):
                raise IndexError(f"fixed index {idx} out of range 0..{max_idx}")

            input_ids, targets = val_loader.get_batch(idx)  # (B, T), (B, T)
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            if targets.dtype != torch.long:
                targets = targets.long()

            logits, _ = model(input_ids)  # (B, T, V)
            V = logits.size(-1)
            if torch.any((targets < 0) | (targets >= V)):
                raise ValueError(f"Targets out of range [0, {V-1}] found in validation batch at index {idx}.")

            if pad_id is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, V),
                    targets.reshape(-1),
                    reduction='sum',
                    ignore_index=pad_id
                )
                valid_tokens = (targets != pad_id).sum().item()
                total_tokens += valid_tokens
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, V),
                    targets.reshape(-1),
                    reduction='sum'
                )
                total_tokens += targets.numel()

            total_nll += loss.item()

    avg_val_loss = total_nll / max(1, total_tokens)
    val_ppl = math.exp(avg_val_loss)
    return avg_val_loss, val_ppl


def save_model(model, optimizer, epoch, loss, save_path):
    checkpoint = {
        'model_config': model.get_model_config(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_class': 'GPT'
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
    print(f"Model config: {checkpoint['model_config']}")


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    return device


