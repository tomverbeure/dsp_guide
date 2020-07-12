
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

# Chapter 3 - ADC and DAC

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

# Chapter 4 - DSP Software

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

# Chapter 5 - Linear Systems

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


# Chapter 6 - Convolution

## The Delta Function and Impulse Response

* backbone of DSP: impulse and Fourier decomposition
* when impulse decomposition is used 

* normalized impulse response: 1 at sample 0, 0 everywhere else -> unit impluse
* impulse response: h[n]

## Convolution

* for a filter, the impulse response is called "filter kernel", "convolution kernel" or "kernel"
* for image processing, it's called "point spread function"
* output signal length = input signal length + impulse response length -1

## The Input Side Algorithm

* x[4]h[n-4]  : the contribution of impulse response for sample 4 = all samples of h, shifted by 4.
* convolution is commmutative
* Center equation of the calculation loop: y[i+j] += x[i] * h[i]

## The Output Side Algorithm

* Center equation of the calculation loop: y[i] += h[j] * x(i-j)
    * Since the math uses y = f(x), the output side equation is what's used in the mathematical
      definition.

* h[n] is flipped left-for-right: 
    * The impulse response describes how each point in the input signal affects the output signal.
    * This results in each point in the output signal being affected by points in the input signal 
      weighted by a flipped impulse response.
* Padding is needed for samples on the left and the right. These zeros can result in bogus results
  on the left and the right.

## Sum of Weighted Inputs

# Chapter 7 - Properties of Convolution

## Common Impulse Responses

* Delta function:
    * delta function: output is identical to input
        * ideal for data storage, communication and measurement
    * k x delta: amplifier or attenuator
    * delta with a shift: delays or advance a signal
    * echo: delta function + scaled and delayed delta function

* Calculus-like operations:
    * first-difference/discrete derivative
        * y[n] = x[n] - x[n-1]
    * running sum/discrete integral
        * step function that extends to infinity
        * recursive
        * y[n] = y[n-1] + x[n]

* Low-pass and high-pass filters:
    * low-pass filter 
        * kernels are composed of adjacent positive points -> averages or smooths the incoming signal
        * may have some negative points as well, e.g. a sinc(x) kernel
        * exponential decay: simplest recursive filter
        * rectangle: best for reducing noise while keeping edge sharpness
        * sinc: used to separate one band of frequencies from another
    * high-pass filter
        * common design method: design low-pass kernel, transform into what you need
            * delta impulse response  -> passes the signal
            * low pass -> only passes low frequence
            * superposition: delta impluse minus low pass -> high pass
            * delta impulse usually added at the center of symmetric coefficients or
              at location 0 if not symmetric
            * sum of all impulse points must be 0 to achieve 0 DC gain
    * cutoff frequency changed by making kernel wider or narrower
    * if low-pass filter has DC gain of 1, then the sum of all impulse response points must be 1

* Causal and Noncausal Signals
    * causal system: all kernel samples before 0 are 0

* Zero phase, linear phase, nonlinear phase
    * zero phase: left-right symmetry around sample 0
    * linear phase: left-right symmetry around some other sample
    * non-linear phase: no left-right symmetry
    * why? Frequency spectrum has magnitude and phase. Spectrum of a symmteric around 0 signal has
      phase 0, etc. (XXX: try this out in Python?)

## Mathematical Properties

* Commutative Property
* Associative Property
* Distributive Property
* Transference between Input and Output: linear operations on x[n] will also happen on y[n]
* Central Limit Theorem: if a pulse-like signal is convolved with itself many times, a Gaussian is produce
    * Width of the Gaussian = original pulse width * sqrt(number of convolutions)
* Correlation
    * cross-correlation: 2 signals correlated into third signal
    * auto-correlation: signals correlated with itself
    * width of the target signal is 2x size of original signal
    * optimal technique for detecting a known waveform in random white noise
        * also called: matched filtering
    * in convolution machine, signal is flipped left-for-right. Not true for correlation machine.

## Speed

* Signal of N samples and signal of M sample -> N x M multiplication for convolution
* Better solution: use FFT.


# Chapter 8 - The Discrete Fourier Transform

## The Family of the Fourier Transform

* Why decompose into sinusoids? Because of sinusoidal fidelity
* Sinusoids are the only waveform that have this property
* 4 kinds of Fourier transforms:
    * Aperiodic-Continuous -> Fourier Transform
    * Periodic-Continuous -> Fourier Series
    * Aperiodic-Discrete -> Discrete Time Fourier Transform
    * Periodic-Discrete -> Discrete Fourier Transform
* There is no Fourier transform that deals with finite length signals. All go from negative to positive infinity
    * The way around this: extend buffer either by zeros (aperiodic) or as periodic repeating.
    * However, aperiodic means infinite number of sinusoids -> only discrete fourier transform can be used for DSP
* Each Fourier transform has real and complex versions
    * Complex version adds a lot of complexity. Only small part of the book touches on this.

## Notation and Format of the Real DFT

* N point input signal (time domain) -> 2 N/2+1 output signals: amplitudes of sines and cosines (freq domain)
    * Forward DFT: from time domain to freq domain
    * Inverse DFT: from freq domain to time domain
* Number of samples in time domain is usual power of 2: efficient for FFT
* Freq domain has 2 parts of N/2+1 values: Re X[n] (cosines) and Im X[n] (sines)


## The Frequency Domain's Independent Variable

* Horizontal axis in freq domain can be referred in 4 different ways. (Example with N=128 time domain samples)
    * 0 to 64: an index into the Re X[n] and Im X[n] array
    * 0 to 0.5: fraction of the sample rate. Divide horizontal axis by N
    * 0 to Pi: multiply fraction by 2Pi. Often used by mathematicians
    * analog frequency: fraction multiplied by sampling frequency. E.g. sampled at 10kHz -> 0 to 5kHz

## DFT Basis Functions

* sine and cosine waves of DFT are basis functions: unity amplitude.
* coefficients ck -> coefficient for cosine basis function that has k complete cycles over the number of samples N.
* Re X[0] holds the DC value.
* Im X[0] value is irrelevant because basis function is zero everywhere.
* cN/2: alternates between -1 and 1
* sN/2: always zero (crossings), just like s0 -> also irrelevant
* A time signal with N samples only has N relevant numbers in the freq domain

## Synthesis, Calculating the Inverse DFT

* Synthesis equation: sum of N+1 weighed cosine and N+1 weighed sine waveforms
    * Weights in the sum are scaled due to some normalization!
        * different for weight c0 and weight cN/2
    * IOW: frequency domain values != sinusoidal weights
    * why? because frequency domain is defined as spectral density
        * freq domain values represent how much signal is present per unit of bandwidth
        * the amount of bandwidth on the left and on the right (is only 1/2 of the ones in the middle)
        * so: middle values -> 2/N bandwidth, left and right: 1/N bandwidth
* an impulse in the time domain -> constant value in the freq domain (XXX: try with Python)

## Analysis, Calculating the DFT

3 different ways to calculate DFT
* DFT by Simultaneous Equation
    * Given N values in time domain, calculate N values in freq domain
    * Set of linear equations
    * Solve, e.g. using Gaussian elimination
    * Never used due to high number of calculations
* DFT by Correlation
    * Multiply input signal samples with correspodning sample of one of the basis function, sum -> coefficient
    * "analysis equation"
    * No special treatment of first and last point needed
    * Requires basis functions to be orthogonal (which they are)
* FFT
    * See later

## Duality

* Analysis (DFT) and synthesis (inverse DFT) equations are strikingly similar: in both cases, known values are
  multiplied with the basis functions and added together
* This symmetry is callded duality
* A single point in the time domain corresponds to a sinusoid in the frequency domain. And a single point in the
  frequency domain corresponds to a sinusoid in the time domain. (XXX: try with Python)
* convolution in the time domain corresponds to multiplication in the freq domain. And convolution in the
  freq domain corresponds to multiplication in the time domain

## Polar Notation

* cosine/sine = rectangular notation: Re X[] and Im X[]
* polar notation: Magnitude of X[] and Phase of X[] (Mag X[] and Phase X[])
* Simple trigonometric conversion function from one notation to the other
* First and last point should alway have Phase == 0.
* rectangular almost always used for mathematical calculations. Polar is often easier to understand
  for humans and graphs

## Polar Nuisances

* radians vs degrees
* divide by zero error when converting from rectangular to polar, due to Im X[]/ Re X[]
* incorrect arctan
* phase of very small magnitudes -> introduces roundoff noise
* 2Pi ambiguitiy of the phase
* magnitude is always positive
* spikes between -Pi and Pi due to noise

# Chapter 9 - Applications of the DFT

## Spectral Analysis of Signals

* DFT with larger number of samples provides better frequency resolution but doesn't reduce
  noise level. Alternative: multiple shorter segments -> shorter DFTs, then average frequency values
* multiply segments with Hamming window
* random noise reduces in proportion to square root of the number of segments.  (XXX: try in Python)
    * Sometimes use millions of segments to bring out weak features
* alternative to reduce noise
    * take very long DFT
    * use low pass digital filter to smooth the spectrum.
        * e.g. average 64 adjacent samples in the original spectrum to create filtered spectrum
* when to use what solution?
    * segments with averaged results is usually the best choice. 

* many measured frequency spectra contain 1/f noise (aka pink noise). Source of it is unknown. 
* what when 2 frequencies close together?
    * sample spacing must be smaller than the distance between the 2 peaks
    * length of the signal limits the frequency resolution: over a short run, 2 sine waves of very close
      frequency added together may look like one signal.
* what if input signal falls between 2 basis functions?
    * it becomes a peak with tails that extend pretty far
    * Hamming window reduces the extent of the tails
        * however, it also reduces the resolution by making all peaks wider.
        * windows create tradeoff between resolution and spectral leakage
* effect of windowing in the frequency domain
    * multiplication in time domain -> convolution in freq domain
    * spectrum of the windowing function is convolved with the original peak
* select N points from signal, but then add an amount of padding zeros beyond the windowing curve (XXX: try with Python)
    * has the effect of sampling the frequency spectrum's continuous curve
* flat-top window: used when amplitude of spectral peak must be measured accurately. 
    * ensures that one or more spectral samples will have the correct peak value
    * but poor frequency resolution
    * shape of flat-top window is exactly the filter kernel of a low-pass filter.

## Frequency Response of Systems

* Fourier transform -> every input signal can be represented as set of cosines with amplitude + phase
* DFT can also be used to represent each output signal in similar form
* Any linear system can be completely described by how it changes amplitude and phase of cosine waves passing
  through it
* This is called the frequency response
* one-to-one relationship between impulse response (time domain) and frequency response (freq domain) through
  the Fourier transform
* time domain: x * h -> y, freq domain: X x H -> Y
* padding impulse response with zeros -> DFT -> higher resolution frequency response
    * nothing limits you from adding more zeros to get higher resolution!
    * even though the impulse response is discrete, the corresponding freq response is continuous

## Convolution via the Frequency Domain

* given impulse response and output signal, what's the input signal?
    * deconvolution: hard in the time domain, easy in freq domain: just divide Y by H
* convolution is mathematically slow, FFT is very fast
* amplitude/phase freq domain convolution is easy: multiply amplitude, add phase.
* rectangular form freq domain convultion: some trig math required
* Convolution with 2^n input samples and some other-length impluse response -> non-2^n output
    * when doing convolution via freq domain -> excess output samples get rolled back beginning of 2^n output
    * distorted output
    * "circular convolution"
    * avoided by padding the input samples with zeros to the next 2^n (XXX: try with Python)
* FFT requires power of 2 nr of samples. Freq domain convolution usually can't guarantee this ->
  circular convolution: distorted version of the correct signal

# Chapter 10 - Fourier Transform Properties


## Linearity of the Fourier Transform

* homogeneity: change in amplitude in one domain produces an identical change in amplitude in the other domain
    * rectangular form: both Re and Im are scaled accordingly
    * polar form: magnitude is scaled accordingly.
* addition: x1[n] + x2[n] = x3[n] -> Re X1[f] + Re X2[f] = Re X3[f], Im X1[f] + Im X2[f] = Im X3[f], 
    * doesn't work with polar, need to transfer back and forth to rectangular format
* Fourier transform is NOT shift invariant

## Characteristics of the Phase

* A shift of s samples in time domain, results in 2Pi x s x f changes in phase
* when signal shifted to the right, phase shows decreasing slope (XXX: Python)
* phase is flat line when timedomain signal is symmetrical and centered around 0
    * Centered around the middle sample is also centered around 0, due to wrap-around in the time domain.
* What happens when things rotate around? What determines if the slope goes upwards/flat/downwards at tipping point?
    * Ambiguitities between Pi, 2Pi, ... (?)
    * See chapter 8. (Polar Nuisances subsection?) 
    * Try this out with Python to really understand.
* Phase shift is proportional to the frequency of the sinusoid being shifted
    * 1 sample shift of low frequency sinusoid -> small phase shift
    * 1 sample shift of highest possible frequency 'sinusoid' -> 180 degree phase shift
* Example on how the phase of the FFT, not the magnitude, encodes where interesting changes happen in the
  time domain.
    * This is becase a rapid change/edge in the time domain requires that all sine waves are changing in the
      same direction at that time. That's determined by the phase.
    * time domain encoding -> phase is most important, freq domain encoding -> magnitude matters (e.g. audio)
* Symmetrical signal in time domain has zero phase across the board.
    * When decomposed into left part (and zeros) and right part (and zeros), you get opposite phase
    * Added together -> they cancel eachother out.
    * flipping a time domain signal from left to right -> magnitude stays the same and phase gets inverted
    * changing the sign of the phase == "complex conjugation", represented with a *
        * X[F] = Mag X[F], Phase X[F], X*[F] = Mag X[F], -Phase X[F]
        * X[F] = Re X[F], Im X[F], X*[F] = Re X[F], -Im X[F]
        * x[n] -> X[F], x[-n] -> X*[F]
        * a[n] * b[n] -> convolution, a[n] * b[-n] -> correlation
        * A[f] x B[f] -> convolution, A[f] x b*[f] -> correlation
        * X[f] x X*[f] -> phase 0 result -> x[n] * x[-n] = left/right symmatrical around sample 0

## Periodic Natures of the FFT

* DFT: periodicity in time domain AND in frequency domain
* DFT assumes period signal -> time domain aliasing
    * transformations done in the frequency domain that are supposed to only impact later samples 
      in the time domain might roll around to earlier samples
    * e.g. circular convolution
* periodicity in the frequency domain -> frequence domain aliasing
* a cosine in time domain maps to +f. -f. and an infinite number of higher fs

<summarize>

## Compression and Expansion, Multirate methods

* Continuous time Fourier Transform: compression in one domain -> expansion in the other domain
    * x(t) -> X(f), then x(kt) -> 1/k x X(f/k)
* DFT: similar, but... 
    * aliasing
    * what was the underlying 'real' signal?
        * depends on whether information was encoded in time domain or freq domain.
        * time domain -> smooth time domain interpolation: linear interpolation, spline interpolation, ...
        * freq domain -> pad freq with zeros -> IDFT.
            * has *exactly* the same frequencies, but time domain signal will have ringing at sharp edges etc.

## Multiplying Signals (Amplitude Modulation)

* convolution in one domain <-> multiplication in the other domain
* AM modulation: multiply in time domain -> convolution in freq domain
* negative frequencies of baseband signal are moved up left of carrier
    * carrier wave, upper sideband, lower sideband

## The Discrete Time Fourier Transform

* operates on aperiodic, discrete signals
    * time domain signal is still discerete
    * frequency domain version is continuous
        * unit: f or w (lower case omega, removes the need of 2pi)
* inverse DTFT: 
    * for IDFT, samples 0 and N/2 divided by 2, not necessary for IDTFT
    * 2/N normalization -> 1/pi normalization
* used for mathematical derivations

## Parseval's Relation

* time domain and freq domain are equivalent representations -> they have the same energy as well
* subtlety when dealing with X[0] and X[N/2]: divide by 1/sqrt(2)
* few practical uses in DSP


# Chapter 11 - Fourier Transform Pairs

* When a simple shape A in time domain -> a simple shape B in freq domain
* Duality: ... and shape A in freq domain -> shape B in time domain
    * E.g. rectangular pulse <-> sync

## Delta Function Pairs

* impluse in time domain 
    * flat magnitude and linear phase in freq domain (mag/phase)
    * sinusoid in freq domain (Re/Im) !
    * location of pulse in time domain determins freq of sinusoid in freq domain

## Sinc Function

* rectangular puse in time domain
    * sinc in freq domain
* wrapped and unwrapped magnitude
    * wrapped: magnitude is always positive -> phase 180 when magnitude was supposed to be negative
    * unwrapped: magnitude can be positive and negative
* sinc doesn't decay to 0 at 0.5 f
    * aliasing...


... a bit more stuff... (TODO)


# Chapter 12 - The Fast Fourier Transform

## Real DFT Using the Complex DFT 

* FFT is based on the complex DFT, but can also be used for real FFT.
* Set all Im parts in the time domain to 0 (duh)
* FFT calculates N points for Re and Im in f domain. Samples 0 to N/2 correspond to spectrum of real DFT
* calculation real IFFT using complex IFFT
    * requires correct loading of negative frequencies

## How the FFT works

## FFT Programs

* In-place computation: same arrays are used for input, intermediate storage, and output
* decimation in frequency vs decimation in time
    * different order in which operations are performed
    * not very important
* Complex DFT has stronger duality between time and freq than real DFT
    * FFT and IFFT are almost identical: just adjust some data at the end

## Speed and Precision Comparisons

* FFT is more precise because less operations -> less rounding errors

## Further Speed Increases

* some optimizations -> between 20% and 40% faster
* stop decomposition 2 stages earlier -> compute freq domain with 4-point sine and cosine
* real FFT: don't calculations that only for imaginary part of time domain

# Chapter 13 - Continuous Signal Processing

Techniques for continuous signal processing nearly identical to DSP. Only theoretical.

## The Delta Function

* Continous signals can be decomposed into scaled and shifted delta functions
* Very short pulses as input to a system 
    * Shape of the input pulse doesn't really matter
    * Shape of the output is determined by the system
    * Area under the pulse determines amplitude
* continuous delta function:
    * infinitesimally brief
    * occurs at time zero
    * area of 1
    * amplitude is infinite, but this doesn't matter.
* output of delta function is the impulse response

## Convolution

* Can be viewed from the input side and from the output side
    * from input side: best to understand conceptually
    * from output side: matches mathematics
* math can be complicated, with a lot of piecemeal sections

## The Fourier Transform

* periodic/aperiodic
    * periodic -> Fourier Series
    * aperiodic -> Fourier Transform

## The Fourier Series

* periodic signals -> frequency spectrum has harmonics
* spectrums can be viewed in 2 ways:
    * continuous, but zero except at the harmonics
    * discrete and only defined at the harmonics
* frequency multiplication
    * Create 10MHz signal
    * distort the sine wave by clipping with a diode -> harmonics
    * bandpass filter
    * sequential stages for doublers and triplers
    * Fourier series describes the amplitude of the multiplied signal depending
      on the selected distortion type and the selected harmonic

# Chapter 14 - Introduction to Digital Filters

## Filter Basics

* separation of signals or restoration of signals
* digital filters 
    * can achieve thousands of times better performance than analog filters
    * performance of filter itself is often ignored (because they're so good), 
      emphasis shifts to limitations of the signal and the theoretical issues
      to process the signal.
* typical input domains: time or space

* every linear filter has: impulse response, step response, frequency response
    * each contains complete information of the filter
* FIR filter: impulse response convolution is used to create a filter, the impulse
  response is also called the filter kernel
* IIR filter: digital filter with recursion
    * filter output is not only a weighed sum of the input, but also has
      weighed sum of previous output values
    * defined by recursion coefficients

* dBV: signal is referenced to a 1 volt RMS signal
* dBm: signal is referenced to signal producing 1mW into a 600 Ohm load (~0.78 V RMS)
* -3dB:  amplitude is reduced to 0.707 and power is reduced to 0.5.

## How Information is Represented in Signals

* information either representated in the time domain or in the frequency domain
    * time domain information -> check step response
    * freq domain information -> check freq response

## Time Domain Parameters

* step response is more useful than impulse response: it matches how humans view information
* parameters:
    * risetime
    * overshoot
    * linear phase

## Frequency Domain Parameters

* passband
* stopband
* transistion band
    *  very narrow -> fast roll-off

* parameters:
    * fast roll-off
    * passband ripple
    * stopband attenuation
    * not: phase
        * usually not very important for freq domain information
        * if necessary, it's easy to make a digital filter with perfect phase response

# High-Pass, Band-Pass and Band-Reject Filters

* Design low pass filter -> convert
* 2 methods to go from low pass to high pass filter
    * spectral inversion
    * spectral reversal

* spectral inversion
    * 



