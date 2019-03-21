import numpy as np
import math
import matplotlib.pyplot as plt
import utils
from obspy import read

# 1 - Read/synthetize data

# Synthetize data
tmax = 20
dt = 0.002
nt = math.floor(tmax/dt) + 1
t = np.arange(0,nt)*dt
f = 1.0

wavelet = utils.ricker(f, dt)
d = np.zeros(nt)
d[round(nt/2)] = 1.0
d = np.convolve(d,wavelet,'same')  #observed data

td = 0.2                        #time delay (if td > 0, syn arrive after obs)
s = utils.shift(d, dt, td)            #synthetic data
s = 0.8*s

# Read data
#st = read('trival_data.sac',debug_headers=True)
#d = st[0].data
#st = read('trival_syn.sac',debug_headers=True)
#s = st[0].data
#dt = st[0].stats.delta
#nt = st[0].stats.npts
#t = np.arange(0,nt)*dt

plt.plot(t,d,'b')
plt.plot(t,s,'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Observed (blue) and synthetic data (red)')
plt.savefig('Figure1.pdf')
plt.show()

# 2 - Butterwoth bandpass filter data (optional)

# 3 - Windowing
n1 = 0
n2 = nt

d = d[n1:n2]    # d[n1] -> d[n2-1]
s = s[n1:n2]    # s[n1] -> s[n2-1]
t = np.arange(n1,n2)*dt # n1*dt -> (n2-1)*dt
nt = len(t)

plt.plot(t,d,'b')
plt.plot(t,s,'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Windowed observed and synthetic data')
plt.show()

# 4 - pre-process/time domain taper applied to windowed data
alpha = 10
nlen = len(s)
it_axis = np.arange(0,nlen)
cos_taper = 1.0 - (np.cos(math.pi * it_axis / (nlen - 1))) ** alpha
d = d * cos_taper
s = s * cos_taper

plt.plot(t, cos_taper)
plt.title('Cosine taper')
plt.show()

plt.plot(t,d,'b')
plt.plot(t,s,'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Tapered observed and synthetic data')
plt.show()

# 5 - Compute_cc
cc = np.correlate(d, s, "same")       # cc is not normalized
ishift = np.argmax(cc) - int(nlen/2)
tshift = ishift*dt
dlnA = 0.5 * math.log(sum(d*d) / sum(s*s) )
print('Time shift measured by cc:', tshift, 's')
print('Amplitude difference measured by cc:', dlnA)

# 6 - Deconstruct_dat_cc (Apply CC -\delta T and -\delta A to the observed data prior to MTM)
# Why?
d_dec = np.zeros(nlen)
for i in range(0, nlen):
    if (i + ishift) > 0 and (i + ishift) < nlen - 1:
        d_dec[i] = d[i + ishift]
if ishift < 0:
    d_dec[0:-ishift] = d_dec[-ishift+1]
if ishift > 0:
    d_dec[nlen-ishift-1:nlen-1] = d_dec[nlen-ishift-2]

d_dec = d_dec * math.exp(-dlnA)

plt.plot(t,s,'r')
plt.plot(t,d_dec,'b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Synthetic data and Deconstruct data')
plt.show()

# 7 - Compute the estimated uncertainty for the cross-correlation measurement
# based on integrated waveform difference between the data and reconstructed synthetics
# compute_average_error

# 8 - FFT parameters
nextpow2 = math.ceil(math.log2(nt))
nfft = 2 * int(math.pow(2, nextpow2))
f0 = 0.0
df = 1.0/(nfft * dt)
dw = 2 * math.pi * df
fnum = int(nfft/2) + 1

fvec = np.zeros(nfft)    # angular frequency vector
fvec[0:fnum] = df * np.arange(0,fnum)    # positive frequency
fvec[fnum:nfft] = df * np.arange(-fnum + 2, 0)  # negative frequency

wvec = np.zeros(nfft)    # angular frequency vector
wvec[0:fnum] = dw * np.arange(0,fnum)    # positive frequency
wvec[fnum:nfft] = dw * np.arange(-fnum + 2, 0)  # negative frequency

# Adjoint source
# {\dot{u}*\delta \tau}/N
v = np.gradient(s)
a = np.gradient(v)
N = sum(a*s)
adj = v * tshift / N
plt.plot(t,adj)
plt.xlabel('Time (s)')
plt.ylabel('Amplitdue')
plt.title('Normalized adjoint source')
plt.savefig('Figure2.pdf')
plt.show()

# Estimate stopping frequency
WTR = 0.02
ampmax = 0.0
k_amp_max = 0
S = np.fft.fft(s,nfft)
D = np.fft.fft(d,nfft)

for k in range(0, fnum):
    if  abs(S[k]) > ampmax:
        ampmax =  abs(S[k])
        k_amp_max = k

wtr_level = ampmax * WTR    # water level value to stablize

fmax = fnum
fmax_stop = 0

for k in range(0, fnum):
    if abs(S[k]) <= wtr_level and fmax_stop == 0 and k > k_amp_max:
        fmax_stop = 1
        fmax = k
    if abs(S[k]) >= 10.0 * wtr_level and fmax_stop == 1 and k > k_amp_max:
        fmax_stop = 0
        fmax = k

print('Stopping frequency index',fmax)
print('Stopping frequency',fmax*df, 'Hz')

#Spectrum of synthetic data
S_SPEC = np.abs(S)
D_SPEC = np.abs(D)
plt.plot(wvec[0:fmax],S_SPEC[0:fmax])
plt.plot(wvec[0:fmax],D_SPEC[0:fmax])
plt.title('Spectrum of synthetic and observed data')
axes = plt.gca()
axes.set_xlim([0,1.2])
plt.show()


