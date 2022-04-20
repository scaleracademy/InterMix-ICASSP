import os
import numpy as np
import random
import chainer
import cmath

import utils as U


class SoundDataset_Train(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, opt, train=True):
        self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.opt = opt
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)//2

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        sound1 = None
        sound2 = None
        label1 = None
        label2 = None
        if self.mix:
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)
            
            signalFFT = np.fft.rfft(sound1)
            signalPhase = np.angle(signalFFT)
            theta1=np.random.uniform(low=-np.pi/2,high=np.pi/2)
            newSignalFFT = signalFFT * cmath.rect( 1., theta1)
            sound1 = np.fft.irfft(newSignalFFT)

            signalFFT = np.fft.rfft(sound2)
            signalPhase = np.angle(signalFFT)
            theta2=np.random.uniform(low=-np.pi/2,high=np.pi/2)
            newSignalFFT = signalFFT * cmath.rect( 1., theta2)
            sound2 = np.fft.irfft(newSignalFFT)

            eye = np.eye(self.opt.nClasses)
            label1 = eye[int(label1)]
            label2 = eye[int(label2)]
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
            theta = np.array([theta1,theta2])

        else:  
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound1 = U.random_gain(6)(sound1).astype(np.float32)
            sound2 = U.random_gain(6)(sound1).astype(np.float32)
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
            theta = np.array([theta1,theta2])


        if self.mix:
            return sound, label, theta

        return sound, label


class SoundDataset_Val(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, opt, train=True):
        self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.opt = opt
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        if self.mix:  
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            signalFFT = np.fft.rfft(sound1)
            signalPhase = np.angle(signalFFT)
            theta1=np.random.uniform(low=-np.pi/2,high=np.pi/2)
            newSignalFFT = signalFFT * cmath.rect( 1., theta1)
            sound1 = np.fft.irfft(newSignalFFT)


            signalFFT = np.fft.rfft(sound2)
            signalPhase = np.angle(signalFFT)
            theta2=np.random.uniform(low=-np.pi/2,high=np.pi/2)
            newSignalFFT = signalFFT * cmath.rect( 1., theta2)
            sound2 = np.fft.irfft(newSignalFFT)

            eye = np.eye(self.opt.nClasses)
            label1 = eye[int(label1)]
            label2 = eye[int(label2)]
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
            theta = np.array([theta1,theta2])

        else:
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound1 = U.random_gain(6)(sound1).astype(np.float32)
            sound2 = U.random_gain(6)(sound1).astype(np.float32)
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
            theta = np.array([theta1,theta2])
        
        if self.mix:
            return sound, label, theta

        return sound, label


def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)), allow_pickle = True)

    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i == split:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    # Iterator setup
    train_data = SoundDataset_Train(train_sounds, train_labels, opt, train=True)
    val_data = SoundDataset_Val(val_sounds, val_labels, opt, train=False)
    train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False)
    val_iter = chainer.iterators.SerialIterator(val_data, 1, repeat=False, shuffle=False)

    return train_iter, val_iter


def setup_timit(opt):
    
    train_dataset = np.load(os.path.join(opt.data, opt.dataset, 'train.npz'), allow_pickle = True)
    test_dataset = np.load(os.path.join(opt.data, opt.dataset, 'test.npz'), allow_pickle = True)
    
    train_sounds = train_dataset["sounds"]
    test_sounds = test_dataset["sounds"]
    train_labels = train_dataset["labels"]
    test_labels = test_dataset["labels"]

    size_train=len(train_labels)
    size_test=len(train_sounds)

    train_data = SoundDataset_Train(train_sounds, train_labels, opt, train=True)
    val_data = SoundDataset_Val(test_sounds, test_labels, opt, train=False)
    train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False)
    val_iter = chainer.iterators.SerialIterator(val_data, 1, repeat=False, shuffle=False)

    return train_iter, val_iter
