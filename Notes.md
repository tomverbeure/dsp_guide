
# The Scientist and Engineer's Guide to Digital Signal Processing

Link: https://dspguide.com

# Chapter 1 - The Breath and Depth of DSP

* A list of different 

# Chapter 2 - Statistics, Probability and Noise

## Signal and Graph Terminology

* continuous vs discrete signals
* x-axis -> independent variable, the domain
* y-axis -> dependent variable, the range
* time domain, frequency domain, spatial domain
* In this book: start sample number at 0

## Mean and Standard Deviation

* average deviation: summation of absolute value of deviation divided by number of samples
    * Almost never used because we care more about the power of a signal
* standard deviation: sqrt(summation of power of 2 of deviation  divided by number of samples-1)
* variance = (standard deviation)^2
* how to calculate variance as a running statistic: updating as new values come in.
* signal-to-noise ratio (SNR) = mean divided by the standard deviation

##  Signal vs Underlying Process

* statistics = what happened, the acquired signals
* probability = understand the underlying process that created the signal

e.g. head or tails: 50% probability, but trying 1000 times won't give you a 500/500 statistic.

* mean and standard deviation are used both for statistics and for probability -> context matters
* histogram and probability mass function are the same thing for statistics and probability resp.

* mean: divide by N, standard deviation divide by N-1. Why?

    * standard deviation relies on mean (which uses N).
    * mean is measured -> has a statistical error due to noise.
        * this error tends to reduce the calculated value of the standard deviation. (Why???)
    * to compensate -> use N-1 instead of N. 
        * doesn't matter for large N
        * makes standard deviation for small N more accurate
    * N-1 is used when estimating the standard deviation of a process by using an acquired signal
    * N is used when calculating the standard deviation of an acquired signal

## The Histogram, Pmf and Pdf

* histogram
* mean and standard deviation can be calculated from histogram.
    * Much faster to calculate
* probability mass function (pmf) is the curve of a process that corresponds to a histogram
    * histogram: finite number of samples
    * pmf: what you would get with infinite number of samples
* y-axis:
    * histogram: number of samples
    * pmf: fraction of samples
* pmf: describes probability that a certain value will be generated
* pmf: only possible for signals with discrete value
* probability density function or pdf: ~pmf but for continuous signals
* to calculate a probability, integrate the pdf over a desired range
* histogram can have bins to reduce number of countable values

## The Normal Distribution

* normal distribution or Gaussian distribution formula
* full formula includes terms for mean, standard deviation and normalization so that area is 1
    * notation: P(x)

* cumulative distribution function (cdf): integral of the pdf, from -infinity to certain value of pdf
    * notation: Phi(x)
    * calculate probabilty of a range by subtracting Phi(top) - Phi(bottom)

## Digital Noise Generation

* need to generate random noise to test algorithms
* sum a certain number of randon numbers -> result has gaussian distribution
* Central Limit Theorem: "a sum of random numbers becomes normally distributed as more and more of the random numbers are added together"
    * random numbers don't need to have any particular distribution!
* alternative to generate gaussian random numbers: get 2 random numbers, use some formula with log(R1) and cos(R2)
* concept of random vs pseudo-random numbers

## Precision and Accuracy

* precision ~standard deviation of measured results
    * precision is a measure of random noise
* accuracy ~distance of mean of measured result from the real value 
    * accuracy is a measure of calibration

# ADC and DAC

## Quantization

* sampling: convert the independent variable from continuous to discrete
* quantization: convert the dependent variable from continuous to discrete
* in most cases, quantization results in nothing more than the addition of a specific amount of random noise
    * mean of zero
    * standard deviation of 1/sqrt(12) * LSB or ~0.29 LSB
* random noise signals are combined by adding their variances
* when deciding to how many bits to digitize: ask how much noise there's already in the system and how much can be tolerated extra
* quantization noise model is not valid when it's not random: e.g. when dealing with a signal that has an amplitude that is close
  to LSB. Or when you have a very slow varying signal so that LSB stays stuck for quite a while.
* dithering: adding noise provides more information!
    * In example, the added noise has a standard deviation of 2/3 LSB. (Why 2/3 ???)
* subtractive dither: create random signals, use DAC to create noise, sample, subtrace random signal.

## The Sampling Theorem

* proper sampling: if you can reconstruct the original signal fromt the samples
* aliasing
    * information can be lost about the higher and the lower frequency.
    * not only different frequency but also 180 degree phase shift
* Shannon or Nyquist sampling theorem: proper sampling -> highest frequency must be 1/2 of sampling rate
* explanation of the effect aliasing by using impulse train
    * series of infinitesimal spikes. Orignal signal is sampled when there is a spike
    * in the time domain, sampling is achieved by multiplying the original signal by an impulse train of unity amplitude spikes.
    * The frequency spectrum of this unity amplitude impulse train is also a unity amplitude impulse train, with the spikes 
      occurring at multiples of the sampling frequency, fs, 2fs, 3fs, 4fs, etc.
    * When two time domain signals are multiplied, their frequency spectra are convolved.
    * This results in the original spectrum being duplicated to the location of each spike in the impulse train's spectrum.

## Digital-to-Analog Conversion

* theoretically perfect: create impulses and a low pass filter
* in practise: zeroth-order hold (~sample and hold of an ADC) -> stair case
    * = convolution of impulses with rectangular pulse
* zeroth-order hold in time domain has sinc(x) function in frequency domain
* spectrum of zeroth-order hold: signal spectrum * sinc(x)
* need reconstruction filter to counteract the sinc(x) and to remove frequencies above 1/2 of sample freq
    * Options:
    * ignore it
    * create analog filter
    * use multirate technique (see later)
    * correction in software before DAC

## Analog Filters for Data Conversion

* full pipeline: analog antialias filter - ADC - DSP - DAC - analog reconstruction filter
* more and more DSP replaces hardware with software
* 3 types of analog filters with different performance parameters
    * Chebyshev
    * Butterworth
    * Bessel
* describes what the filter does, not how it is implemented
* common building block for analog filter: modified Sallen-Key circuit
    * [Analysis of the Sallen-Key Architecture](https://www.ti.com/lit/an/sloa024b/sloa024b.pdf)
* switched cap analog filter: replaces resistors by switched capacitors
    * benefit: easy to produce in silicon
    * cut-off frequency is directly proportional to the switching frequency
        * great for acquisition systems with different sample rate
* performance parameters
    * cutoff frequency sharpness: Chebyshev > Butterworth > Bessel
    * passband ripple
        * Chebyshev has a ripple of ~6% (0.5dB), a good compromise and a common choice
        * elliptical filter allows ripple in passband and stopband for an even better tradeoff between
          roll-off and passband ripple
        * Butterworth: designed for maximum rolloff without passband ripple -> maximally flat filter
    * step response
        * Butterworth and Chebyshev -> overshoot and ringing
        * Bessel -> no overshoot, no ringing. 
        * Bessel: "linear phase":  rising and falling edge of output look similar.
* the frequency band between 0.4 and 0.5 of the sampling frequency is a wasteland of filter roll-off and antialiased signals

## Selecting The Antialias Filter

* Filter selection depends on how information is represented in the signals you intend to process
* frequency domain encoding 
    * shape of the waveform not important, only frequency.
    * Sharp cut-off needed. 
    * step response not important
    * Chebyshe/Elliptic/Butterworth
* time domain encoding
    * shape of waveform used to store information
    * sampling theorem doesn't help in understanding how a time domain signal should be digitized.
    * Bessel filter is probably best
    * Set cutoff frequency at around 25% of sample frequency
    * Use more poles for sharper cutoff
    * around 2 samples for the rising portion of each edge

## Multirate Data Conversion

* Use sample rate much faster than dictated by sample theorem
    * Makes antialiasing and reconstruction filter much easier.
* multirate system uses more than 1 sample rate
    * ADC 
        * simple analog RC filter
        * ADC: sample at very high rate
        * use DSP to filter undesired frequencies
        * decimate: resample to lower sample frequency
    * DAC
        * interpolate (insert zeros between impulses)
        * digital low pass filter
        * DAC
        * simple analog RC filter

## Single Bit Data Conversion

* trade off high sampling rate / lower number of bits
* delta modulation
* 3 example circuits
    * basic delta modulator
        * positive/negative charge injector
        * relative number of ones/zeros defines the slope
        * all bits have the same meaning, no MSB or LSB etc.
        * requires very high bit rate to reach acceptable slew rate/quantization error
    * continuous variable slope delta modulator
        * changes the amount of charge injected when consecutive zeros or ones
        * syllabic filter: characteristics depends on the average length of the syllables making up speech
        * easiest way to transmit voice, quality not great, no really usable for anything else
    * delta-sigma convertor
        * Contrary to before, the comparator creates a counteracting charge to keep a capacitor at zero
        * relative number of zeros and ones in the level of the incoming voltage, not the slope
        * output can easily be converted back into analog with an analog low pass filter (e.g. RC network)
        * easy to decimate
        * limitations: 
            * difficult to multiplex inputs
            * unclear *when* a sample is taken, since it's done over multiple clocks. Difficult to use for
              time domain processing.

# DSP Software

## Computer Numbers

* 2 prevalent types: fixed point and floating points

## Fixed Point (Integers)

* unsigned, offset binary, sign and magnitude, two's complement

## Floating Point (Real Numbers)

## Number Precision

* quantization is constant for fixed point, variable for floating point
* rounding errors can cause gradual drift of the result

## Execution Speed: Program Language

* assembly, compiler, interpreter ...

## Execution Speed: Hardware

* discusses CPU, cache, math coprocessor, memory from a 1996 point of view...

## Execution Speed: Programming Tips

(a lot of outdated stuff as well...)

* use integer instead of floating point
* avoid complex functions (sin, cos, ...), use Taylor or Maclaurin power series instead
* use LUTs
* profile your code

# Linear Systems

## Signals and Systems

* continuous systems: x(t), y(t)
* discrete systems: x[t], y[t]
* most useful system are linear

## Requirements for Linearity

* linear -> homogeneity and adititivy
    * homogeneity: x[n] -> y[n] then kx[n] -> ky[n]
* another essential property for DSP: shift invariant

## Static Linearity and Sinusoidal Fidelity

* static linearity and sinusoidal fidelity help with intuitive understanding of what a linear system is.
* static linearity: the output is the input multiplied by a constant
* sinusoidal fidelity: If the input to a linear system is a sinusoidal wave, the output will also be a 
  sinusoidal wave, and at exactly the same frequency as the input.
    * sinusoids are the only waveforms with this characteristic.

## Examples of Linear and Nonlinear Systems

...

## Special Properties of Linearity

* commutative: 2 linear systems can be cascaded one way or the other, the final output is the same.
* multiplication is linear as long as one of the inputs is a constant

## Superposition: the Foundation of DSP

* synthesis: the process of combining signals through scaling and addition
* decomposition: the inverse operation where a signal is broken up into two or more additive components
* superposition: the overall strategy for understanding how signals and systems can be analyzed
    * e.g. a signal x[n] can be split into different impulse signals: input signal components)
    * these are then sent through the system: output signal components
    * added back together: synthesized into y[n]

## Common Decompositions

* impulse decomposition
* step decomposition
* even/odd decomposition
    * even signal: where y[n] is mirrored around N/2
    * odd signal: where y[n] is the opposite around N/2
* circular symmetry
    * the end of a signal is linked to the beginning of the signal
    * important for Fourier analysis
* interlaced decomposition
    * even samples with zeros in between, odd samples with zeros in between
    * used to calculate FFTs
* Fourier decomposition
    * N-point signal is decomposed into N+2 signals: half sine waves and half cosine waves
    * the only thing different for different inputs are different amplitudes
    * basis for Fourier analysis, Laplace and z-transforms

## Alternatives to Linearity

* only one way major strategy to deal with non-linear systems: make them resemble a linear system
* how?
    * ignore the non-linearity
    * keep signals small (e.g. transistors)
    * apply linearizing transform / homomorphic transformation 
        * e.g. instead of a = b * c -> log(a) = log(b) + log(c)


# Convolution

## The Delta Function and Impulse Response

* backbone of DSP: impulse and Fourier decomposition
* when impulse decomposition is used 

