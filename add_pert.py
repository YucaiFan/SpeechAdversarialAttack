import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile

import scipy.io.wavfile as wav

import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

import DeepSpeech

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits
    
    
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--test_path', type=str,
                    required=True,
                    help="Test dir")
parser.add_argument('--pert', type=str,
                    required=True,
                    help="Pert path")
parser.add_argument('--out_path', type=str,
                    required=True,
                    help="Out dir")
args = parser.parse_args()


with tf.Session() as sess:
    audios = []
    lengths = []
    pert = np.load(args.pert)
    i = 0

    if(args.test_path):
        for f in os.listdir(args.test_path):
            if f.split(".")[-1] == 'mp3':
                raw = pydub.AudioSegment.from_mp3(args.test_path + "/" + f).set_frame_rate(16000)
                audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
            elif f.split(".")[-1] == 'wav':
                fs, audio = wav.read(args.test_path+"/"+f)
                assert fs == 16000
                assert audio.dtype == np.int16
            l = len(audio)
            x_test = list(audio)
            maxlen_test = max(len(pert),len(x_test))
            original_test = np.array(x_test+[0]*(maxlen_test-len(x_test)))
            final_pert = list(pert)
            final_pert = np.array(final_pert+[0]*(maxlen_test-len(final_pert)))
            rescale_test = np.max(np.abs(final_pert))/2000.0
            apply_delta_test = tf.clip_by_value(final_pert, -2000, 2000)*rescale_test
            apply_delta_test = list(apply_delta_test.eval())
            mask_test = np.array([1 if i < l else 0 for i in range(maxlen_test)])
            test_adv = apply_delta_test*mask_test + original_test
            path = args.out_path+"out_"+str(i)+".wav"
            i += 1
            wav.write(path, 16000,
              np.array(np.clip(np.round(test_adv[:l]),
                               -2**15, 2**15-1),dtype=np.int16))
