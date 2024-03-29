# -*- coding: utf8 -*-
import numpy as np
#from read import *
from PIL import Image

def calc_stft(signal, frame_length=200, frame_step=52, winfunc=np.hamming, NFFT=255):

    # Calculate the number of frames from the signal
    frame_length = frame_length
    frame_step = frame_step
    signal_length = len(signal) #selected as 6855
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    #print(frames.shape)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    # pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)

    return mag_frames #pow_frames

def normalize(image):

    mean = np.mean(image)
    var = np.mean(np.square(image-mean))

    image = (image - mean)/np.sqrt(var)

    return image


if __name__ == '__main__':
    import scipy.io.wavfile
    import matplotlib.pyplot as plt

    # Read wav file
    # "OSR_us_000_0010_8k.wav" is downloaded from http://www.voiptroubleshooter.com/open_speech/american.html
    sample_rate, signal = scipy.io.wavfile.read("./OSR_us_000_0010_8k.wav")
    #signal=data_c[2]
    # Get speech data in the first 2 seconds
    #signal = signal[0:int(2. * sample_rate)]
    #signal = data_c[2][0,6800*4:6800*5]
    #signal=signal[0:6800]
    # Calculate the short time fourier transform
    pow_spec = calc_stft(signal)
    print(pow_spec.shape)
    plt.axis('off')
    plt.imshow(pow_spec, aspect='equal')
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.show()
    # name是保存路径
    #plt.savefig(name, bbox_inches='tight', pad_inches=0)

    #plt.show()
    plt.savefig('test_1.png',bbox_inches='tight', pad_inches=0)
    # img=Image.fromarray(np.uint8(pow_spec))
    # img.show()