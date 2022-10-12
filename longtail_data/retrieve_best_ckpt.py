import imp


import argparse
import json

def main(args):
    path = args.ckpt_path
    with open(path) as f:
        data = json.loads(path)
        dir = data['best_model_checkpoint']
    return dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path')
    main(parser.parse_args())