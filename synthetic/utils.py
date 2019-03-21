import math
import cmath
import numpy as np

def ricker(f, dt):
    """
        Ricker wavelet of central frequency f
        
        IN    f : central frequency in Hz
        dt : time sampling interval in sec
        
        OUT   the Ricker wavelet
        
        Example
        wavelet = ricker(10, 0.002)
        
        Credit: Ke Chen, kechen@lbl.gov
        """
    
    tw = 2 * 1.35 * math.sqrt(6.0)/math.pi/f
    
    nw = 2 * math.floor(tw/dt/2.0) + 1
    
    nc = math.floor(nw/2)
    
    k = np.arange(1,nw + 1)
    
    alpha = (nc - k + 1)*dt*f*math.pi
    
    beta = alpha*alpha
    
    w = (1.0 - 2.0*beta)*np.exp(-beta)
    
    return w


def shift(d, dt, delay):
    """
        Shift a signal by time delay
        """
    nt = len(d)
    nextpow2 = math.ceil(math.log2(nt))
    nfft = int(math.pow(2, nextpow2))
    
    D = np.fft.fft(d,nfft)
    S = np.zeros(nfft, dtype=complex)
    
    for i in range(0, round(nfft/2) + 1):
        w = 2.0*math.pi*i/nfft/dt
        Shift = cmath.exp(-1j*w*delay)
        S[i] = D[i]*Shift
    
    # frequency domain symmetries
    for i in range(1,round(nfft/2)):
        S[nfft - i] = np.conjugate(S[i])
    
    s = np.fft.ifft(S)
    s = np.real(s[0:nt])
    return s
