# -*- coding: utf-8 -*-
  
"""
Created on Mon Nov 10 17:32:14 2022

@author: Yacine Benkirane
Contributions, worked with various classmates on the discord (From Recollection): 
    Matt, Nic, Simon, Guillaume, Jeff, Liam, Louis, Chris, Steve
"""

import numpy as np
import readligo
import h5py
import sys
import scipy
import json
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100

data_dir="./LOSC_Event_tutorial/"
sys.path.append(data_dir)

json_fname = "BBH_events_v3.json"
events = ['GW151226','GW170104', 'GW150914','LVT151012']



def Convolution(u,v):
    
    u_ft=np.fft.fft(u)
    v_ft=np.fft.fft(v)
    
    conv = np.fft.ifft(u_ft*v_ft)
    
    return np.fft.fftshift(conv) 

def Gauss(x, mu, sig):
    
    return np.exp(-0.5 * (x - mu)**2/sig**2)

def retrieveSpectrum(time, strn, window, smooth_width):
    
    frqs = np.fft.fftfreq(len(time),(time[1] - time[0]))
    
    ps = np.abs(np.fft.fft(strn*window))**2
    
    funSmooth = Gauss(frqs,0,smooth_width)
    
    return np.fft.fftshift(frqs), np.fft.fftshift(ps), Convolution(ps,funSmooth/funSmooth.sum())

def filterMatch(strn,window,template,noise_model):
    
    ftData = np.fft.fft(window * strn)
    
    ftTempl = np.fft.fft(window * template)
    
    rhs = np.fft.ifft(np.conj(ftTempl)*np.reciprocal(noise_model)*ftData)
    
    return np.fft.fftshift(rhs)  


def readFile(data_dir, fileString):
    
    data = h5py.File(data_dir + fileString,'r')
    temp = data['template']
    return temp[0], temp[1]

def SN1(filterMatch,window,template,noise_model):
    
    noise = np.sqrt(np.mean(filterMatch**2))  
    return np.abs(filterMatch)/noise

def findEvents(json_fname, events):
    
    #Was very confused about this for quite some time
    
    events = json.load(open(data_dir + json_fname, 'r'))
    times, frqs, ps, smoothPS, filterMatches, SN, templates = {}
    
    for event in events:
    
        event = events[event]   
        fn_H1 = event['fn_H1']
        fn_L1 = event['fn_L1']
        tevent = event['tevent']
        
        strn_Ha, time_Ha, chan_dict_Ha = readligo.loaddata(data_dir+fn_H1,'H1')
        strn_Li, time_Li, chan_dict_Li = readligo.loaddata(data_dir+fn_L1,'L1')
        
        time = time_Ha - tevent
        template = readFile(data_dir,event['fn_template'])[0]
        
        window = scipy.signal.tukey(len(time), 0.1)
        
        frqs, ps_Ha,smoothPS_Ha = retrieveSpectrum(time,strn_Ha,window,2)
        noise_model_Ha = np.fft.fftshift(smoothPS_Ha)
        
        filterMatch_Ha = filterMatch(strn_Ha,window,template,noise_model_Ha)
        SN1_Ha = SN1(filterMatch_Ha,window,template,noise_model_Ha)
        
        frqs,ps_Li,smoothPS_Li = retrieveSpectrum(time,strn_Li,window,2)
        noiseLi = np.fft.fftshift(smoothPS_Li)
        
        filterMatch_Li = filterMatch(strn_Li,window,template,noiseLi)
        SN1_Li = SN1(filterMatch_Li, window, template, noiseLi)

        times[event] = time
        
        ps[event] = ps_Ha, ps_Li
        
        smoothPS[event] = smoothPS_Ha, smoothPS_Li
        
        filterMatches[event] = filterMatch_Ha,filterMatch_Li
        
        SN[event] = SN1_Ha,SN1_Li
        
        templates[event] = template
        
    return times,frqs,ps,smoothPS,filterMatches,SN,templates

times, frqs, ps, smoothPS, filterMatches, SN, templates = findEvents(json_fname,events)

for event in events:
    
    plt.loglog(frqs,ps[event][0])
    plt.loglog(frqs,smoothPS[event][0])
    plt.xlim([0.1,2500])
    plt.ylim([10**(-45), 10**(-29)])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Intensity')
    plt.grid()
    plt.title(event + ': Hartsford Detector')
    plt.legend(['Raw Spectrum','Smoothed Spectrum'])
    plt.savefig('PSET6_Q5_1_'+ event +'.png',bbox_inches='tight')
    plt.show()
    
    plt.loglog(frqs, ps[event][1])
    plt.loglog(frqs, smoothPS[event][1])
    plt.xlim([0.1,2500])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Intensity')
    plt.ylim([10**(-45),10**(-29)])
    plt.title(event + ': Livingston Detector')
    plt.grid()
    plt.legend(['Raw Spectrum','Smoothed Spectrum'])
    plt.savefig('PSET6_Q5_2_'+ event +'.png',bbox_inches='tight')
    plt.show()
                                        
    
eventExample = 'LVT151012'    #Just serves as an example 
    
plt.plot(times[eventExample], filterMatches[eventExample][0])
plt.xlabel('Time Offset (sec)')
plt.grid()
plt.title('LVT151012: Matched Filter Hartsford')
plt.savefig('PSET6_Q5_3Harts.png',bbox_inches='tight')
plt.show()

plt.plot(times[eventExample], SN[eventExample][0])
plt.xlabel('Time Offset (sec)')
plt.grid()
plt.title('LVT151012: SNR Hartsford')
plt.savefig('PSET6_Q5_4Harts.png',bbox_inches='tight')
plt.show()

plt.plot(times[eventExample], filterMatches[eventExample][1])
plt.xlabel('Time Offset (sec)')
plt.grid()
plt.title('LVT151012: Matched Filter Livingston ')
plt.savefig('PSET6_Q5_3Liv.png',bbox_inches='tight')
plt.show()

plt.plot(times[eventExample], SN[eventExample][1])
plt.xlabel('Time Offset (sec)')
plt.grid()
plt.title('LVT151012: SNR Livingston')
plt.savefig('PSET6_Q5_4Liv.png',bbox_inches='tight')
plt.show()

for event in events:
    plt.plot(times[event],filterMatches[event][0])
    plt.plot(times[event],filterMatches[event][1])
    plt.xlim([-15,15])
    plt.grid()
    plt.legend(['Hanford Detector', 'Livingston Detector'])
    plt.xlabel('Time Offset (sec)')
    plt.title(event + ' Matched Filter Comparison')
    plt.savefig('PSET6_Q5_MatchedFilter_'+ event+'.png',bbox_inches='tight')
    plt.show()


for event in events:
    plt.plot(times[event],SN[event][0])
    plt.plot(times[event],SN[event][1])
    plt.xlim([-10,10])
    plt.grid()
    plt.legend(['Hanford Detector', 'Livingston Detector'])
    plt.xlabel('Time Offset (sec)')
    plt.title(event + ' SNR Comparison')
    plt.savefig('PSET6_Q5_SNR_'+ event+'.png', bbox_inches='tight')
    plt.show()

for event in events:
    time = times[event]
    n = np.argmax(SN[event][0])
    m = np.argmax(SN[event][1])
    print(event + ' Time Difference in sec', np.abs(time[n] - time[m]))

