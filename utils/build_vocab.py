import argparse
import os
import tokenizers

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vocabulary Generator')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input *.txt file or directory of *.txt files '
                             'containing sentences')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output vocab file')
    parser.add_argument('-s', '--size', type=int, default=30000,
                        help='Vocab size')
    parser.add_argument('--tokenizer', type=str, default='wordpiece',
                        choices=['wordpiece', 'bpe'],
                        help='Tokenization Method')
    parser.add_argument('--uppercase', action='store_true', default=False,
                        help='Build a cased vocab')
    parser.add_argument('--special_tokens', nargs='+', 
                        default=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
                        help='Build a cased vocab')
    parser.add_argument('--pad_token', type=str, default='[PAD]',
                        help='Padding token (will be given index 0)')
    args = parser.parse_args()

    input_files = []
    if os.path.isfile(args.input):
        input_files.append(args.input)
    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.txt'):
            if path.is_file():
                input_files.append(str(path))
    else:
        raise ValueError("{} is not a valid path".format(args.input))

    if args.tokenizer == 'wordpiece':
        tokenizer = tokenizers.BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            lowercase=not args.uppercase,
        )
    elif args.tokenizer == 'bpe':
        tokenizer = tokenizers.ByteLevelBPETokenizer(
            add_prefix_space=True,
            lowercase=not args.uppercase,
            trim_offsets=True,
        )

    print('Starting training', flush=True)
    tokenizer.train(
        input_files,
        vocab_size=args.size,
        show_progress=True,
        special_tokens=args.special_tokens,
    )
    print('Finished training', flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # save the vocab
    vocab = [(word, freq) for word, freq in tokenizer.get_vocab().items()]
    vocab.sort(key=lambda x: x[1])
    vocab = [word for (word, freq) in vocab]

    # Force special tokens to front of list
    for token in args.special_tokens:
        vocab.insert(0, vocab.pop(vocab.index(token)))

    # Force padding token to be index 0
    if args.pad_token in vocab:
        vocab.pop(vocab.index(args.pad_token))
    vocab.insert(0, args.pad_token)

    with open(args.output, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')
    print('Vocab written to file', flush=True)

