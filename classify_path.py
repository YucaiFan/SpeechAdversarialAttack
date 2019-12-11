## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav

import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")
import DeepSpeech

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

from tf_logits import get_logits


# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"



def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_path', type=str, required=False, help="Input folder path")
    parser.add_argument('--in', type=str, dest="input",
                        required=False,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    args = parser.parse_args()
    audios = []
    lengths = []

    while len(sys.argv) > 1:
        sys.argv.pop()
    with tf.Session() as sess:
        

        if args.input_path:
            for f in os.listdir(args.input_path):
                if f.split(".")[-1] == 'mp3':
                    raw = pydub.AudioSegment.from_mp3(args.input_path+"/"+f)
                    raw = raw.set_frame_rate(16000)
                    audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
                elif f.split(".")[-1] == 'wav':
                    fs, audio = wav.read(args.input_path+"/"+f)
                audios.append(list(audio))
                lengths.append(len(audio))
        else:
            if args.input.split(".")[-1] == 'mp3':
                raw = pydub.AudioSegment.from_mp3(args.input)
                raw = raw.set_frame_rate(16000)
                audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
                print(">> hahaha")
            elif args.input.split(".")[-1] == 'wav':
                _, audio = wav.read(args.input)
            else:
                raise Exception("Unknown file format")
            audios.append(list(audio))
            lengths.append(len(audio))

        N = len(audios)
        maxlen = max(map(len, audios))
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        #finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])

        for i in range(len(audios)):
            audio = audios[i]
            new_input = tf.placeholder(tf.float32, [1, N])
            new_length = tf.placeholder(tf.int32, [1])

            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                logits = get_logits(new_input, new_length)

            saver = tf.train.Saver()
            saver.restore(sess, args.restore_path)

            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, new_length, merge_repeated=False, beam_width=500)

            print('logits shape', logits.shape)
            lengths = (len(audios[i])-1)//320
            print(">> lengths:", lengths)
            #length = (len(audio)-1)//320
            l = len(audio)
            r = sess.run(decoded, {new_input: [audios[i]],
                                   lengths: [length]})


            print("-"*80)
            print("-"*80)

            print("Classification:")
            print("".join([toks[x] for x in r[0].values]))
            print("-"*80)
            print("-"*80)

main()
