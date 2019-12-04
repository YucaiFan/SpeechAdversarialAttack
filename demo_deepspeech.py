import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time
import cv2

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
toks = " abcdefghijklmnopqrstuvwxyz'-"


if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


from targeted_universal_pert import targeted_perturbation
from util_univ import *

# if you want using cpu. change for device='/cpu:0'
#device = '/gpu:0'
device = '/cpu:0'

# choose your target class number based on imagenets or None(Non-target).
#
target = "Winner winner chicken dinner"

def jacobian(y_flat, x, inds):
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=2),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < 2,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()    

    # Default values
    # path_train_imagenet = '/datasets2/ILSVRC2012/train'
    # path_test_image = 'data/test_img.png'

    #path_train_imagenet = '/datasets2/ILSVRC2012/train'
    path_test_audio = 'data/test_audio.wav'


    with tf.device(device):
        persisted_sess = tf.Session()

        
        # Get audio matrix from input audio file:
        if args.input.split(".")[-1] == 'mp3':
            raw = pydub.AudioSegment.from_mp3(args.input)
            audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
        elif args.input.split(".")[-1] == 'wav':
            _, audio = wav.read(args.input)
        else:
            raise Exception("Unknown file format, only support mp3 and wav")
       

        N = len(audio)
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits(new_input, lengths)

        saver = tf.train.Saver()
        saver.restore(persisted_sess, args.restore_path)

        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

        print('logits shape', logits.shape)
        length = (len(audio)-1)//320
        l = len(audio)
        r = persisted_sess.run(decoded, {new_input: [audio],
                               lengths: [length]})


        print("-"*80)
        print("-"*80)

        print("Classification:")
        print("".join([toks[x] for x in r[0].values]))
        print("-"*80)
        print("-"*80)

        print("Finished!")


        def f(audio_inp, length_inp): 
            r_out = persisted_sess.run(decoded, {new_input: [audio_inp], lengths: [length_inp]})
            return "".join([toks[x] for x in r_out[0].values])
        
        print("Now we try f!!!!\nClassification Result:") 
        print(f(audio, (len(audio)-1)//320))
        # deepspeech_model_path = os.path.join('data', 'tensorflow_deepspeech_graph.pb')

        # if os.path.isfile(deepspeech_model_path) == 0:
        #     print("="*20 + " ERROR! " + "="*20)
        #     print('>> Model does not exist!...')

        # model = os.path.join(inception_model_path)
        # model = os.path.join(deepspeech_model_path)

        # Load the Inception model
        # with gfile.FastGFile(model, 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     persisted_sess.graph.as_default()
        #     tf.import_graph_def(graph_def, name='')
            

        
        # =============================
        # Print checkpoint's graph layer name and tensor name
        # =============================
        print(">> Printing names for each layer!")
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        with open('result.txt', 'w+') as f:
            for tensor_name in tensor_name_list:
                f.write(tensor_name+'\n')
        print(">> Layer name in result.txt!")                
        # for op in tf.get_default_graph().get_operations():
        #     print(op.name, op.values())
        # Printing name finished


        # persisted_sess.graph.get_operations()

        # Get deepspeech's input tensor and output tensor!
        persisted_input = persisted_sess.graph.get_tensor_by_name("Placeholder:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("logits:0")
        # need to modify

        print('>> Computing feedforward function...')
        # def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})
        
        # ================================
        # If we have precomputed npy file:
        # ================================
        # if target==None:
        #     file_perturbation = os.path.join('data','precomputing_perturbations', 'universal-nontarget.npy')
        # else:
        #     file_perturbation = os.path.join('data','precomputing_perturbations', 'universal-target-'+str(target).zfill(5)+'.npy')

        # file_perturbation = os.path.join('data', 'pert_M.npy')

        # TODO: Optimize this construction part!
        print('>> Compiling the gradient tensorflow functions. This might take some time...')
        y_flat = tf.reshape(persisted_output, (-1,))
        inds = tf.placeholder(tf.int32, shape=(2,))
        dydx = jacobian(y_flat,persisted_input,inds)

        print('>> Computing gradient function...')
        def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

        #if os.path.isfile(file_perturbation) == 0:
        if True:
            # Load/Create data
            # datafile = os.path.join('data', 'imagenet_data.npy')
            datafile = os.path.join('M.npy')
            if os.path.isfile(datafile) == 0:
                print('>> Creating pre-processed imagenet data...')
                X = create_imagenet_npy(path_train_imagenet)

                print('>> Saving the pre-processed imagenet data')
                if not os.path.exists('data'):
                    os.makedirs('data')

                # Save the pre-processed images
                # Caution: This can take take a lot of space. Comment this part to discard saving.
                np.save(os.path.join('data', 'imagenet_data.npy'), X)

            else:
                print('>> Pre-processed imagenet data detected')
                X = np.load(datafile)

            # Running targeted universal perturbation

            v = targeted_perturbation(X, f, grad_fs, delta=0.25, max_iter_uni=10,target=target)

            # Saving the universal perturbation
            file_perturbation = os.path.join('data','precomputing_perturbations', 'universal-target-'+str(target).zfill(5)+'.npy')
            np.save(os.path.join(file_perturbation), v)

        else:
            print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        print('>> Testing the targeted universal perturbation on an image')

        # Test the perturbation on the image
        # labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')


        test_audio = "test.wav"
        path_test_audio = "../M/" + test_audio

        audio_original = pseudo.read(audio)

        audio_original = pseudo.preprocess(audio_original)

        print("audio_original:", audio_original.shape)

        # image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")
        str_label_original = audio2str(f=f,audio=audio_original)

        # Clip the perturbation to make sure images fit in uint8
        # print(v)

        # print(image_original)

        #image_perturbed = avg_add_clip_pert(image_original,v)
        audio_perturbed = audio_original + v

        pseudo.save(audio_original, "audio_original.wav") #post-processing needed
        pseudo.save(audio_perturbed, "audio_perturbed.wav")
        # print(image_perturbed)
        # label_perturbed = np.argmax(f(audio_perturbed), axis=1).flatten()
        str_label_perturbed = audio2str(f=f, audio=audio_perturbed)

        print(audio_perturbed.shape)
        print(str_label_original, str_label_perturbed)

        # Show original and perturbed image
        # TODO: need to revert average manipulation


