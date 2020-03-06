from pathlib import Path
import argparse

from tokenizers import ByteLevelBPETokenizer

def run_tokenizer(args):
    paths = [str(x) for x in Path(args.input_data_dir).glob("**/*.txt")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save(args.output_dir, args.model_name)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="The name of the model.")
    parser.add_argument("--input_data_dir", default=None, type=str, required=True, help="The input training data directory (*.txt files).")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the tokenizer outputs will be written.")

    args = parser.parse_args()

    run_tokenizer(args)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()