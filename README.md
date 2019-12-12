# SpeechAdeversarialAttack

鸡你太美

1. Install the dependencies

```
pip3 install tensorflow-gpu==1.14 progressbar numpy scipy pandas python_speech_features tables attrdict pyxdg

pip3 install $(python3 util/taskcluster.py --decoder)
```

Download and install
https://git-lfs.github.com/

1b. Make sure you have installed git lfs. Otherwise later steps will mysteriously fail.

2. Clone the Mozilla DeepSpeech repository into a folder called DeepSpeech:

```
git clone https://github.com/mozilla/DeepSpeech.git
```

2b. Checkout the correct version of the code:

```
(cd DeepSpeech; git checkout tags/v0.4.1)
```

2c. If you get an error with tflite_convert, comment out DeepSpeech.py Line 21
```
# from tensorflow.contrib.lite.python import tflite_convert
```

3. Download the DeepSpeech model

```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz
tar -xzf deepspeech-0.4.1-checkpoint.tar.gz
```

4. Verify that you have a file 
```
md5sum ./DS_model/deepspeech-0.4.1-checkpoint/model.v0.4.1.data-00000-of-00001
```
Its MD5 sum should be
ca825ad95066b10f5e080db8cb24b165

5. Check that you can classify normal images correctly

```
python3 attack.py --in sample-000000.wav --restore_path ./DS_model/deepspeech-0.4.1-checkpoint/model.v0.4.1
```

6. Generate adversarial examples

```
python3 attack.py --in sample-000000.wav --target "this is a test" --out adv.wav --iterations 1000 --restore_path ./DS_model/deepspeech-0.4.1-checkpoint/model.v0.4.1
```

8. Verify the attack succeeded

```
python3 classify.py --in adv.wav --restore_path ./DS_model/deepspeech-0.4.1-checkpoint/model.v0.4.1
# should get 'this is a test'

python3 classify.py --in sample-000000.wav --restore_path ./DS_model/deepspeech-0.4.1-checkpoint/model.v0.4.1
# should get something else
```
