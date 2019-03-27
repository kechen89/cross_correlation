from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

#Loop over stations
for file in glob.glob('obs/*.BHE.sac'):
    stname = re.search('BK.(.+?).BHE',file).group(1)
    #read observed data
    st = read(file)
    st += read(file.replace('BHE','BHN', 1))
    st += read(file.replace('BHE','BHZ', 1))
    #print(st)
    #st.plot(size=(800,600))

    #filter observed data
    st2 = st.copy()
    st2.filter("lowpass",freq=0.25,corners=2,zerophase=True)
    #st2.plot(size=(800,600))

    #read simuated data
    file2 = file.replace('BK','BK.BH', 1)
    file3 = file2.replace('BHE.sac','e', 1)
    file4 = file3.replace('obs','syn', 1)
    st3 = read(file4)
    st3 += read(file4.replace('.e','.n', 1))
    st3 += read(file4.replace('.e','.u', 1))

    #st3.filter("lowpass",freq=0.25,corners=2,zerophase=True)
    #st3.plot(size=(800,600))
    c = ['E component', 'N component', 'Z component']
    plt.rcParams["figure.figsize"] = [11,11]
    for i in range(3):
        plt.subplot(3,1,i+1, title=c[i])
        t = np.arange(0, st2[i].stats.npts/st2[i].stats.sampling_rate, st2[i].stats.delta)
        plt.plot(t, st2[i], label='Observed data',color='b')
        t = np.arange(0, st3[i].stats.npts/st3[i].stats.sampling_rate, st3[i].stats.delta)
        plt.plot(t, st3[i], label='Simulated data',color='r')
        plt.legend(loc='upper right')
    plt.suptitle(stname)
    file5 = file.replace('BHE.sac','seismograms.pdf')
    print(file4)
    fig = file5.replace('obs','fig')
    plt.savefig(fig)
    plt.show()

    for tr in st2:
        tr.write(tr.id + "_filtered.SAC", format="SAC")
