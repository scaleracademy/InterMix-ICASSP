# InterMix: An Interference-based Data Augmentation And Regularization Technique For Automatic Deep Sound Classification

Implementation of InterMix: An Interference-based Data Augmentation And Regularization Technique For Automatic Deep Sound Classification by Ramit Sawhney, and Atula Tejaswi Neerkaje.
## Environment & Installation Steps
Python 3.8 & Chainer 7.7.0

## Run

Execute the following steps in the same environment:

```bash
python3 main.py --data data --dataset [DATASET] --mixup_type sound --netType envnetv2 --batchSize 32 --BC --eligible 1 2 3 4 --strongAugment
```

## Command Line Arguments

To run different variants of InterMix, perform ablation or tune hyperparameters, the following command-line arguments may be used:

```
  --dataset DATASET     dataset from ['esc10', 'esc50', 'urbansound8k']
  --mixup-type TYPE     sound (p-weighting) vs. normal
  --bc                  perform mixup
  --eligible L          eligible layer set 
  --strongAugment       perform scale and gain augmentation
  --batchSize BATCH_SIZE
                        batch size
  --nEpochs EPOCHS      number of epochs
  --LR RATE             learning rate
  --weightDecay WD      weight decay
  --momentum MOMENTUM   LR momentum
  --split SPLIT         choice of split
```

## Datasets

Dataset preparation for ESC-50, ESC-10, and UrbanSound8K

- FFmpeg should be installed.
- First of all, please make a directory to save datasets.

		mkdir [path]

##### [ESC-50 and ESC-10](https://github.com/karoldvl/ESC-50) setup

	python esc_gen.py [path]

- Following files will be generated.
	- [path]/esc50/wav16.npz  # 16kHz, for EnvNet
	- [path]/esc50/wav44.npz  # 44.1kHz, for EnvNet-v2
	- [path]/esc10/wav16.npz
	- [path]/esc10/wav44.npz

##### [UrbanSound8K](http://urbansounddataset.weebly.com/urbansound8k.html) setup

1. Download UrbanSound8K dataset from [this page](http://urbansounddataset.weebly.com/urbansound8k.html).

2. Move UrbanSound8K directory.

		mkdir -p [path]/urbansound8k
		mv UrbanSound8K [path]/urbansound8k/
		
3. Run the following command.

		python urbansound_gen.py [path]
		
- Following files will be generated.
	- [path]/urbansound8k/wav16.npz
	- [path]/urbansound8k/wav44.npz

## Cite

If our work was helpful in your research, please kindly cite this work:

```
@inproceedings{wisdom2021s,
  title={InterMix: An Interference-based Data Augmentation And Regularization Technique For Automatic Deep Sound Classification},
  author={Sawhney, Ramit and 
          Neerkaje, Atula Tejaswi},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  organization={IEEE}
}

```

#### References
<i id=1></i>[1] Jindal, A., Ranganatha, N. E., Didolkar, A., Chowdhury, A. G., Jin, D., Sawhney, R., & Shah, R. R. (2020, January). SpeechMix-Augmenting Deep Sound Recognition Using Hidden Space Interpolations. In INTERSPEECH (pp. 861-865).

<i id=2></i>[2] Tokozume, Y., Ushiku, Y., & Harada, T. (2018, February). Learning from Between-class Examples for Deep Sound Recognition. In International Conference on Learning Representations.
