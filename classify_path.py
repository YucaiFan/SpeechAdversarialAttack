import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import fileinput


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_path', type=str, required=False, help="Input folder path")
    parser.add_argument('--out', type=str, dest="output",
                        required=False,
                        help="Output audio .txt file")
    parser.add_argument('--in', type=str, dest="input",
                        required=False,
                        help="Input audio .wav file, at 16KHz (separated by spaces)")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    args = parser.parse_args()

    while len(sys.argv) > 1:
        sys.argv.pop()

    #with open("./test_result.txt", mode="w") as test_result:
    with open(args.output, mode="w") as test_result:
    
        if args.input_path:
            for f in os.listdir(args.input_path):
                cmdstr = "python3.5 classify.py --in "+ os.path.join(args.input_path, f) +" --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1"
                print(cmdstr)
                resstr = os.popen(cmdstr).read()
                print(resstr.split("Classification:")[-1])
                test_result.write(f + " " + resstr.split("Classification:\n")[-1])

main()
