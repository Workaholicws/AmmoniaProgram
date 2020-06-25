# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 5 14:02:20 2016
    
    @author: shenwang
"""

"""
    This py is used to read fits data and give the Tk as results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyfits
import math
from astropy.io import fits

##++++++++++++++++++++++++++++++++++++++++++++++++++++
#  read fits
##++++++++++++++++++++++++++++++++++++++++++++++++++++
###=============
#  Add nh3_11 data way
hdulist11 = pyfits.open('/Users/Desktop/NH3/data/nh3_11.fits')
###=============
spec_nh11 = hdulist11[0].data
header11=hdulist11[0].header

crvalx11 = hdulist11[0].header['crval1']
cdeltax11 = hdulist11[0].header['cdelt1']
crpixx11 = hdulist11[0].header['crpix1']
crvaly11 = hdulist11[0].header['crval2']
cdeltay11 = hdulist11[0].header['cdelt2']
crpixy11 = hdulist11[0].header['crpix2']
crvalz11 = hdulist11[0].header['crval3']
cdeltaz11 = hdulist11[0].header['cdelt3']
crpixz11 = hdulist11[0].header['crpix3']

nx11 = hdulist11[0].header['naxis1']
ny11 = hdulist11[0].header['naxis2']
nz11 = hdulist11[0].header['naxis3']

x11 = np.arange(-crpixx11*cdeltax11+crvalx11,(nx11-1-crpixx11)*cdeltax11+crvalx11,cdeltax11)
y11 = np.arange(-crpixy11*cdeltay11+crvaly11,(ny11-1-crpixy11)*cdeltay11+crvaly11,cdeltay11)
z11 = np.arange(-crpixz11*cdeltaz11+crvalz11,(nz11-1-crpixz11)*cdeltaz11+crvalz11,cdeltaz11)
z11 = z11/1000.0

###=============
#  Add nh3_22 data way
hdulist22 = pyfits.open('/Users/Desktop/NH3/data/nh3_22.fits')
###=============
spec_nh22 = hdulist22[0].data
header22=hdulist22[0].header
crvalx22 = hdulist22[0].header['crval1']
cdeltax22 = hdulist22[0].header['cdelt1']
crpixx22 = hdulist22[0].header['crpix1']
crvaly22 = hdulist22[0].header['crval2']
cdeltay22 = hdulist22[0].header['cdelt2']
crpixy22 = hdulist22[0].header['crpix2']
crvalz22 = hdulist22[0].header['crval3']
cdeltaz22 = hdulist22[0].header['cdelt3']
crpixz22 = hdulist22[0].header['crpix3']

nx22 = hdulist22[0].header['naxis1']
ny22 = hdulist22[0].header['naxis2']
nz22 = hdulist22[0].header['naxis3']

x22 = np.linspace(-crpixx22*cdeltax22+crvalx22,(nx22-1-crpixx22)*cdeltax22+crvalx22,nx22)
y22 = np.linspace(-crpixy22*cdeltay22+crvaly22,(ny22-1-crpixy22)*cdeltay22+crvaly22,ny22)
z22 = np.linspace(-crpixz22*cdeltaz22+crvalz22,(nz22-1-crpixz22)*cdeltaz22+crvalz22,nz22)
z22 = z22/1000.0

##++++++++++++++++++++++++++++++++++++++++++++++++++++
#   funC11() give a easy to use C11 mode in our paper;
#   funC22() give a easy to use C22 mode in our paper. 
##++++++++++++++++++++++++++++++++++++++++++++++++++++
def funC11(dv, x):
    dv=int(10*dv)/10.0
    ###=============     
    if dv==0:
        dv==0.1  #keep min at least 0.1
    ###=============           
    if dv==0.1:
        p=[2.70,-2.78,1.25,0.16,1.96];
    elif dv== 0.2:
        p=[1.19,-0.92,0.68,0.22,1.45];
    elif dv== 0.3:
        p=[0.30,0.34,0.22,0.27,1.14];
    elif dv== 0.4:
        p=[0.32,0.25,0.27,0.29,1.06];
    elif dv== 0.5:
        p=[0.39,0.12,0.33,0.29,1.06];
    elif dv== 0.6:
        p=[0.46,-0.01,0.38,0.28,1.08];
    elif dv== 0.7:
        p=[0.53,-0.11,0.42,0.28,1.10];
    elif dv== 0.8:
        p=[0.58,-0.20,0.46,0.28,1.12];
    elif dv== 0.9:
        p=[0.63,-0.27,0.49,0.27,1.13];
    elif dv== 1.0:
        p=[0.67,-0.33,0.51,0.27,1.14];
    elif dv== 1.1:
        p=[0.70,-0.38,0.53,0.27,1.16];
    elif dv== 1.2:
        p=[0.72,-0.42,0.54,0.27,1.17];
    elif dv== 1.3:
        p=[0.74,-0.45,0.56,0.27,1.17];
    elif dv== 1.4:
        p=[0.76,-0.48,0.57,0.26,1.18];
    elif dv== 1.5:
        p=[0.78,-0.50,0.58,0.26,1.19];
    elif dv== 1.6:
        p=[0.79,-0.52,0.59,0.26,1.19];
    elif dv== 1.7:
        p=[0.80,-0.54,0.59,0.26,1.19];
    elif dv== 1.8:
        p=[0.81,-0.55,0.60,0.26,1.20];
    elif dv== 1.9:
        p=[0.82,-0.58,0.61,0.26,1.20];
    else :
        p=[0.82,-0.58,0.61,0.26,1.20];    
    a1, a2, a3, a4, a5 = p
    y=a1*x*x+a2*x+a3+a4*np.exp(a5*x)
    return y
    
def funC22(dv, x):
#    dv=int(10*dv)/10.0
    if dv==0.0:
        dv==0.1  #keep min at least 0.1
    
    if dv==0.1:
        pp=[5.537,0.837];
    elif dv== 0.2:
        pp=[5.549,0.839];
    elif dv== 0.3:
        pp=[5.530,0.836];
    elif dv== 0.4:
        pp=[5.442,0.827];
    elif dv== 0.5:
        pp=[5.342,0.817];
    elif dv== 0.6:
        pp=[5.264,0.808];
    elif dv== 0.7:
        pp=[5.209,0.802];
    elif dv== 0.8:
        pp=[5.171,0.798];
    elif dv== 0.9:
        pp=[5.144,0.795];
    elif dv== 1.0:
        pp=[5.124,0.793];
    elif dv== 1.1:
        pp=[5.110,0.791];
    elif dv== 1.2:
        pp=[5.099,0.790];
    elif dv== 1.3:
        pp=[5.090,0.789];
    elif dv== 1.4:
        pp=[5.083,0.788];
    elif dv== 1.5:
        pp=[5.078,0.788];
    elif dv== 1.6:
        pp=[5.073,0.787];
    elif dv== 1.7:
        pp=[5.069,0.787];
    elif dv== 1.8:
        pp=[5.066,0.787];
    elif dv== 1.9:
        pp=[5.064,0.786];
    else :
        pp=[5.061,0.786];
    b1, b2 = pp
    y=b1*x**(b2)
    return y
    
##++++++++++++++++++++++++++++++++++++++++++++++++++++
#   input 
##++++++++++++++++++++++++++++++++++++++++++++++++++++    
sigma11=0.008
sigma22=0.008
msigma11=3*sigma11
msigma22=3*sigma22

#dv_22=1.0
ns=2.0
cw11=abs(cdeltaz11/1000)
cw22=abs(cdeltaz22/1000)

n80=0
Tkkfits=np.empty((ny11,nx11))
Tkk=[]

##++++++++++++++++++++++++++++++++++++++++++++++++++++
#  caculate one by one
##++++++++++++++++++++++++++++++++++++++++++++++++++++
for ii in range(nx11):
    for jj in range(ny11):
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  these parameters can be changed by user
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++
        ###  Add spectures ### 
        T11=np.reshape(spec_nh11[0,:,jj,ii],63)
        T22=np.reshape(spec_nh22[0,:,jj,ii],63)
        
        T11_new=np.reshape(spec_nh11[0,5:58,jj,ii],53)
        T22_new=np.reshape(spec_nh22[0,5:58,jj,ii],53)
        z11_new=z11[5:58]
        z22_new=z22[5:58]
        
        if (np.max(T11)>msigma11)&(np.max(T22)>msigma11)&(not math.isnan(T11[0]))&(not math.isnan(T22[0])):            
            midline11=z11_new[np.argwhere(T11_new==np.max(T11_new))]
#            print midline11
            midline22=z22_new[np.argwhere(T22_new==np.max(T22_new))]
            
            z11=z11-midline11[0]
            z22=z22-midline22[0]
            vs_11=z11
            vs_22=z22
            
            num=np.shape(np.argwhere(T22_new>3*sigma22))[0]
            if num==0:
                dv_22=round((np.sqrt((cw11*0.5/np.sqrt(np.log(2)))*(cw11*0.5/np.sqrt(np.log(2)))-0.04)),1)
            else:
                dv_22=round((np.sqrt((cw11*num*0.5/np.sqrt(np.log(2)))*(cw11*num*0.5/np.sqrt(np.log(2)))-0.04)),1)
            
##++++++++++++++++++++++++++++++++++++++++++++++++++++           
#            if dv_22>2.0:
#                dv_22=2.0
##++++++++++++++++++++++++++++++++++++++++++++++++++++       
#            print("dv_22",dv_22)
            max_m=np.max(T22)
            
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++
            #   PS:most of time we don't need os1&os2, to speed up, we annotate it
            ##++++++++++++++++++++++++++++++++++++++++++++++++++++
            #nh3_11 spectures
            indxos1=np.argwhere((vs_11<(-19.4948+ns*dv_22))&(vs_11>(-19.4948-ns*dv_22))&(T11>msigma11))
            indxis1=np.argwhere((vs_11<(-7.4903+ns*dv_22))&(vs_11>(-7.4903-ns*dv_22))&(T11>msigma11))
            indxm=np.argwhere((vs_11<(0.00136+ns*dv_22))&(vs_11>(0.00136-ns*dv_22))&(T11>msigma11))
            indxis2=np.argwhere((vs_11<(7.59233+ns*dv_22))&(vs_11>(7.59233-ns*dv_22))&(T11>msigma11))
            indxos2=np.argwhere((vs_11<(19.5022+ns*dv_22))&(vs_11>(19.5022-ns*dv_22))&(T11>msigma11))
            #nh3_22 spectures
            indxs1=np.argwhere((vs_22<(-16.3917+ns*dv_22))&(vs_22>(-16.3917-ns*dv_22))&(T22>msigma22))
            indxmm=np.argwhere((vs_22<(0.00054+ns*dv_22))&(vs_22>(0.00054-ns*dv_22))&(T22>msigma22))  
            indxs2=np.argwhere((vs_22<(16.3822+ns*dv_22))&(vs_22>(16.3822-ns*dv_22))&(T22>msigma22))    

            if indxis1.shape[0]*indxm.shape[0]*indxis2.shape[0]>0:
                max_is1=np.max(T11[indxis1])
                max_m=np.max(T11[indxm])
                max_is2=np.max(T11[indxis2])
                
                Rs11=np.sum(T11[indxis1]*cw11)+np.sum(T11[indxis2]*cw11)
                Rm11=np.sum(T11[indxm]*cw11)
                Rsm11=(Rs11*1.0)/(Rm11*1.0)
#                if Rsm11<0.5556:
#                    Rsm11=0.5556
                C11=funC11(dv_22,Rsm11)
#                print("C11",C11)
                
                if indxmm.shape[0]>0:
                    Rs22=np.sum(T22[indxs1]*cw22)+np.sum(T22[indxs2]*cw22)
                    Rm22=np.sum(T22[indxmm]*cw22)
#                    R2m=np.sum(T22[indx2m]*cw22)
                    Rsm22=(Rs22*1.0)/(Rm22*1.0)
                    if Rsm22<0.1302:
                        Rsm22=0.1302
    #                    print("Rsm22",Rsm22)
                    C22=funC22(dv_22,Rsm22)
    #                    print("c22",C22)
                    
                    FT_12=((Rs11+Rm11)*1.0)/(Rm22*1.0)
                    cf=C11/C22*FT_12
    #               cf=C11*FT_12
    #                    print("dv_22",dv_22)
    #                    print("C11",C11)
    #                    print("c22",C22)
    #                    print("cf",cf)
    #                    
                    Trot=40.99/np.log(0.811*cf)
                    Tk=3.67+0.307*Trot+0.0357*Trot*Trot
    
                    if (Tk<80)&(Tk>0):
                        Tkk.append(Tk)
                        Tkkfits[jj,ii]=Tk
    #                    print Tk
                    else:
                        n80=n80+1
plt.hist(Tkk, 40, normed=0, facecolor='green', alpha=0.5)
#plt.savefig('/Users/genius/Desktop/work/Tk_1.eps',dpi=100)
print n80
plt.xlabel("Tk(K)")
plt.ylabel("N")
plt.show()
#fits.writeto("/Users/Desktop/nh3_Tk.fits",Tkkfits)
