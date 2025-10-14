import os
from collections import defaultdict
import numpy as np

class Dictionary:
    """
    Manages a bidirectional vocabulary mapping between words and unique indices.

    Attributes:
        word2idx (dict): Maps each word (str) to its unique index (int).
        idx2word (list): Stores words such that idx2word[index] returns the word.

    Example:
        >>> dictionary = Dictionary()
        >>> idx = dictionary.add_word("hello")  # adds "hello" and returns its index
        >>> print(dictionary.word2idx["hello"])  # prints the index for "hello"
        >>> print(dictionary.idx2word[idx])      # prints "hello"
        >>> print(len(dictionary))               # prints vocabulary size (e.g., 1)
    """

    def __init__(self):
        """
        Initializes an empty Dictionary.
        """
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Adds a word to the dictionary if it doesn't exist and returns its index.

        Args:
            word (str): The word to add.

        Returns:
            int: The index of the word in the dictionary.

        Notes:
            - If the word already exists, its existing index is returned.
            - New words are assigned indices in the order they first appear.
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.

        Returns:
            int: Vocabulary size.
        """
        return len(self.idx2word)


class Corpus:
    """
    Processes a text file and converts it into a sequence (list) of integer token IDs.

    The class reads a single text file, builds a vocabulary using the Dictionary class,
    and converts the text into numerical indices based on that vocabulary.

    Attributes:
        dictionary (Dictionary): The vocabulary manager for the corpus.
        data (list[int]): Sequence (list) of token IDs representing the input text.

    Args:
        file_path (str): Directory path containing the text file.
        file_name (str): File name (e.g., 'train.txt').

    Example:
        >>> corpus = Corpus("./data", "train.txt")
        >>> print(corpus.data[:10])             # prints first 10 token IDs
        >>> print(len(corpus.dictionary))       # prints vocabulary size
        >>> # Lookup examples:
        >>> w = corpus.dictionary.idx2word[0]   # word at index 0
        >>> i = corpus.dictionary.word2idx[w]   # index for that word
    """

    def __init__(self, file_path, file_name):
        """
        Initializes the Corpus by creating a dictionary and processing the input file.

        Args:
            file_path (str): Directory path containing the text files.
            file_name (str): File name (e.g., 'train.txt').
        """
        self.dictionary = Dictionary()
        full_path = os.path.join(file_path, file_name)
        self.data = self.tokenize(full_path)

    def tokenize(self, path):
        """
        Tokenizes a text file and converts it to word indices.

        Args:
            path (str): Path to the text file to be tokenized.

        Returns:
            list[int]: List of word indices representing the text.

        Raises:
            AssertionError: If the specified path doesn't exist.

        Notes:
            - Adds '<eos>' token after each line to mark end of sentence.
            - Updates the dictionary with new words as they are encountered.
            - Converts all words to their corresponding indices.
        """
        assert os.path.exists(path), f"File not found: {path}"
        tokens = []

        # Read file line by line, split on whitespace, and append '<eos>' per line
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
                    tokens.append(word)

        # Convert words to their indices
        ids = [self.dictionary.word2idx[word] for word in tokens]
        return ids


def preview_ptb_file(file_path, file_name, num_lines=5):
    """
    Previews the first few lines of a Penn Treebank (PTB) dataset file.

    Opens and displays the contents of the specified file from the PTB dataset,
    showing a limited number of lines for quick inspection.

    Args:
        file_path (str): Directory path containing the file.
        file_name (str): Name of the file to preview (e.g., "train.txt", "valid.txt", "test.txt").
        num_lines (int, optional): Number of lines to display. Defaults to 5.

    Example:
        >>> preview_ptb_file('ptb/data_raw', "train.txt", num_lines=3)
        Previewing ptb/data_raw/train.txt:
        aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ...
        pierre <unk> N years old will join the board as a nonexecutive director nov. N
        mr. <unk> is chairman of <unk> n.v. the dutch publishing group

    Notes:
        - Each line is printed with trailing whitespace removed using rstrip().
        - The preview stops after reaching the specified number of lines.
        - Lines are printed as they appear in the file, preserving original formatting.
    """
    path = os.path.join(file_path, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\nPreviewing {path}:")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if num_lines is not None and (i + 1) >= num_lines:
                break

class MarkovChainLanguageModel:
    """
    An n-gram (order-n) Markov Chain language model that learns next-token
    probabilities from a tokenized corpus and supports generation and prediction.

    The model estimates, for each observed state (a tuple of the previous `order`
    token IDs), a probability distribution over the next token. It can:
      - generate text by sampling successive tokens from these distributions, and
      - predict the most likely next word for a given context.

    Attributes:
        corpus_ids: Sequence (list) of token IDs used to train the model.
        order : Markov order (number of preceding tokens that define a state).
        idx2word: Index-to-word mapping (idx2word[idx] -> word).
        word2idx: Word-to-index mapping (word2idx[word] -> idx).
        transition_model:
            Maps each observed state (tuple of length `order`) to a probability vector
            over the vocabulary. The vector is aligned with `vocabulary` by index.
        vocabulary:
            Sorted list of unique token IDs seen in either state positions or as next tokens.
        index_to_state: List of all observed states in the corpus.

    Example:
        >>> corpus = Corpus("./data", "train.txt")
        >>> model = MarkovChainLanguageModel(corpus=corpus, order=3)
        >>> txt = model.generate_text(length=30)
        >>> pred = model.predict_next_word("the market")
    """

    def __init__(self, corpus=None, order=1):
        """
        Initialize the Markov Chain Language Model from a Corpus.

        Args:
            corpus (Corpus): A Corpus instance with:
                - corpus.data: sequence (list) of token IDs
                - corpus.dictionary.idx2word: list mapping indices to words
                - corpus.dictionary.word2idx: dict mapping words to indices
            order (int, optional): Markov order (context length). Defaults to 1.

        Notes:
            - If `corpus` is provided, the transition model is computed immediately.
            - `order` must be >= 1 and smaller than the effective length of `corpus.data`
              to observe at least one transition.
        """
        self.corpus_ids = corpus.data        
        self.order = order
        self.idx2word = corpus.dictionary.idx2word
        self.word2idx = corpus.dictionary.word2idx
        self.transition_model = None
        self.vocabulary = None
        if self.corpus_ids is not None:
            self.transition_model, self.vocabulary, self.index_to_state = self.calculate_transition_model()


    def calculate_transition_model(self):
        """
        Build the transition model from the training corpus.

        Returns:
            tuple:
                - transition_matrix:
                    For each observed state, a probability vector (numpy array)
                    over `vocabulary`, summing to 1. The i-th entry corresponds
                    to token ID `vocabulary[i]`.
                - vocabulary: Sorted unique token IDs observed in states
                    or as next tokens.
                - index_to_state: All observed states.

        Notes:
            - Counts transitions from each length-`order` state to the following token,
              then normalizes counts to probabilities.
            - Ensures the vocabulary includes tokens appearing either in states or
              as next tokens, so probability vectors align correctly.
        """
        ids = self.corpus_ids
        transition_counts = defaultdict(lambda: defaultdict(int))

        # Count transitions for sequences of `order` tokens
        for i in range(len(ids) - self.order):
            current_state = tuple(ids[i:i + self.order])
            next_token = ids[i + self.order]
            transition_counts[current_state][next_token] += 1

        # Build vocabulary as set of unique token IDs (from states and next tokens)
        all_token_ids = set()
        for current_state in transition_counts:
            all_token_ids.update(current_state)
            all_token_ids.update(transition_counts[current_state].keys())

        vocabulary = sorted(list(all_token_ids))
        transition_matrix = {}
        tokid_to_index = {tokid: idx for idx, tokid in enumerate(vocabulary)}

        for current_state, next_tokens in transition_counts.items():
            total_transitions = sum(next_tokens.values())
            probabilities = np.zeros(len(vocabulary))
            for next_token, count in next_tokens.items():
                next_index = tokid_to_index[next_token]
                probabilities[next_index] = count / total_transitions
            transition_matrix[current_state] = probabilities

        return transition_matrix, vocabulary, list(transition_counts.keys())

    def generate_text(self, length=50, trigger_word=None):
        """
        Generate a sequence of words by sampling from the learned transition model.

        Args:
            length (int, optional): Total number of tokens to produce. Defaults to 50.
            trigger_word (str, optional): If provided, randomly choose an initial
                observed state that contains this word; otherwise start from a random state.

        Returns:
            str: Generated text as a single space-separated string.

        Raises:
            ValueError:
                - If trigger_word is provided but not found in the vocabulary.
                - If no observed state contains the trigger_word.

        Notes:
            - Sampling uses numpy.random.choice with the state's probability vector.
            - If generation reaches an unseen state (no outgoing distribution),
              the method jumps to a random observed state (or a random one containing
              the trigger word, if specified) to continue.
        """
        if trigger_word is None:
            # Choose a random initial state
            current_state = self.index_to_state[np.random.choice(len(self.index_to_state))]
            generated_tokens = list(current_state)
        else:
            # Handle trigger word case
            if trigger_word not in self.word2idx:
                raise ValueError(f"Trigger word '{trigger_word}' not in vocabulary")
                
            trigger_id = self.word2idx[trigger_word]
            
            # Find valid states containing the trigger word
            valid_states = [state for state in self.index_to_state 
                            if trigger_id in state]
            
            if not valid_states:
                raise ValueError(f"No valid states found containing '{trigger_word}'")
                
            # Choose a random state containing the trigger word
            current_state = valid_states[np.random.choice(len(valid_states))]
            generated_tokens = list(current_state)

        # Generate remaining tokens
        for _ in range(length - self.order):
            if current_state in self.transition_model:
                probabilities = self.transition_model[current_state]
                next_token_idx = np.random.choice(len(self.vocabulary), p=probabilities)
                next_token = self.vocabulary[next_token_idx]
                generated_tokens.append(next_token)
                current_state = tuple(generated_tokens[-self.order:])
            else:
                # If current state not found, choose a new random state
                if trigger_word is None:
                    current_state = self.index_to_state[np.random.choice(len(self.index_to_state))]
                else:
                    current_state = valid_states[np.random.choice(len(valid_states))]
                generated_tokens.extend(list(current_state))

        # Convert token IDs back to words
        words = [self.idx2word[token] for token in generated_tokens]
        return ' '.join(words)
    
    def predict_next_word(self, query, order=None):
        """
        Predict the most probable next word for a given context.

        Args:
            query (str | list[int]):
                - A whitespace-delimited string of words, or
                - A list of token IDs.
            order (int, optional): Markov order to use for this prediction.
                Defaults to the model's `order`.

        Returns:
            str | None: The most likely next word, or None if the state is unseen
            or the query contains out-of-vocabulary words.

        Raises:
            ValueError: If the query contains fewer than `order` tokens.

        Notes:
            - For string input, words not in the vocabulary cause the method to
              return None (with a message).
            - If the derived state is not present in the transition model,
              the method returns None (with a message).
        """
        if order is None:
            order = self.order
        if isinstance(query, str):
            # Tokenize using word2idx
            words = query.strip().split()
            tokens = []
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    print(f"Word '{w}' not in vocabulary. Returning None.")
                    return None
        else:
            tokens = query
        if len(tokens) < order:
            raise ValueError(f"Query must contain at least {order} tokens.")
        query_state = tuple(tokens[-order:])
        if query_state not in self.transition_model:
            print(f"State '{query_state}' is not in the transition model")
            return None

        next_word_distribution = self.transition_model[query_state]
        max_prob_token = self.vocabulary[np.argmax(next_word_distribution)]
        return self.idx2word[max_prob_token]

    def sample_next_word(self, query, order=None):
        """
        Sample a next word from the state's probability distribution.

        Args:
            query (str | list[int]):
                - A whitespace-delimited string of words, or
                - A list of token IDs.
            order (int, optional): Markov order to use for this prediction.
                Defaults to the model's `order`.

        Returns:
            str | None: A sampled next word, or None if the state is unseen
            or the query contains out-of-vocabulary words.

        Raises:
            ValueError: If the query contains fewer than `order` tokens.

        Example:
            >>> model.sample_next_word("the cat")  # Random sampling based on probabilities

        Notes:
            - For string input, words not in the vocabulary cause the method to
              return None (with a message).
            - If the derived state is not present in the transition model,
              the method returns None (with a message).
        """
        # Use default order if not specified
        if order is None:
            order = self.order

        # Convert string query to tokens if needed
        if isinstance(query, str):
            words = query.strip().split()
            tokens = []
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    print(f"Word '{w}' not in vocabulary. Returning None.")
                    return None
        else:
            tokens = query

        # Check query length
        if len(tokens) < order:
            raise ValueError(f"Query must contain at least {order} tokens.")

        # Get the state and check if it exists in model
        query_state = tuple(tokens[-order:])
        if query_state not in self.transition_model:
            print(f"State '{query_state}' is not in the transition model")
            return None

        # Get probability distribution
        probabilities = self.transition_model[query_state]

        # Sample next token using probabilities
        next_token = np.random.choice(self.vocabulary, p=probabilities)
        return self.idx2word[next_token]

