from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import pylab as pl
from numpy import *

def main():
  print 
  print "Enter 1 for 2D thick; 2 for 2D freq; 3 for 2D ang; 4 for 3D thick vs ang; "
  choice=input("5 for 3D thick vs freq; 6 for freq vs ang; 7 for 3D n vs k; 8 for 3D n vs k thick: ")
  if choice==1:
    thick_2D()
  elif choice==2:
    freq_2D()
  elif choice==3:
    ang_2D()    
  elif choice==4:
    thick_ang_3D()
  elif choice==5:
    thick_freq_3D()
  elif choice==6:
    freq_ang_3D()
  elif choice==7:
    n_k_3D()
  elif choice==8:
    n_k_thick_3D()
 
#=============================================================================================
  
def thick_2D():
 
  nlayer=1 ; ang1=25.0 # deg
  freq=0.5 # THz
  thick1=2.0 # mm
  npt=1000 # total data points
  
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  thick=linspace(0.0,thick1,npt) ; temp=zeros(nlayer)
  irad=zeros(npt,complex) ; Rte=zeros(npt,complex) ; Rtm=zeros(npt,complex)
  n[0]=1.0+0j ; n[1]=4.5-0.2j ; n[2]=1+0j
  ang=ang1*pi/180. 
  
  Rtm0,Rte0,irad0=Rair_hs(ang,n[1])
  print 'half space',irad0

  for i in xrange(npt):
    temp[0]=thick[i]
    Rtm[i],Rte[i],irad[i] = Rlayer(nlayer,n,temp,ang,freq)
    # print Rtm,Rte,irad[i]
      
  pl.plot(thick, irad, 'b-',linewidth=1.0)
  pl.plot(thick, abs(Rte),'g--', linewidth=1.0)
  pl.plot(thick, abs(Rtm), 'g-.',linewidth=1.0)
  pl.legend( ('Irradiance','TE Refl. Coef','TM Refl. Coef'), loc='lower right')
  pl.xlim(0.0,thick1)
  pl.ylim(0.0,1.0)
  pl.xlabel('Layer Thickness (mm)')
  pl.ylabel('Magnitude')
  # pl.title('8 THz; 25 deg.; n,k=4.5,0.2 in air ')
  pl.grid(True)
  pl.show()

#=============================================================================================
  
def freq_2D():
 
  nlayer=1 ; npt=500 ; ang1=25.0 # deg
  thick=0.01 # mm
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  
  freq=linspace(0.0,15.0,npt) ; temp=zeros(nlayer)
  irad=zeros(npt,complex) ; Rte=zeros(npt,complex) ; Rtm=zeros(npt,complex)
  n[0]=1.0+0j ; n[1]=4.5-0.2j ; n[2]=1+0j
  ang=ang1*pi/180.

  for i in xrange(npt):
    temp[0]=thick
    Rtm[i],Rte[i],irad[i] = Rlayer(nlayer,n,temp,ang,freq[i])
    
  pl.plot(freq, irad, 'b-',linewidth=1.0)
  pl.plot(freq, abs(Rte),'g--', linewidth=1.0)
  pl.plot(freq, abs(Rtm), 'g-.',linewidth=1.0)
  pl.legend( ('Irradiance','TE Refl. Coef','TM Refl. Coef'), loc='upper right')
  pl.xlim(0.0,15.0)
  pl.ylim(0.0,1.2)
  pl.xlabel('Frequency (THz)')
  pl.ylabel('Magnitude')
  # pl.title('8 THz; 25 deg.; n,k=4.5,0.2 in air ')
  pl.grid(True)
  pl.show()

#=============================================================================================
  
def ang_2D():
 
  nlayer=1 ; npt=91 ; freq=8.0 #THz
  thick=0.05 # mm
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  
  ang1=linspace(0.0,90.0,npt) ; temp=zeros(nlayer)
  irad=zeros(npt,complex) ; Rte=zeros(npt,complex) ; Rtm=zeros(npt,complex)
  n[0]=1.0+0j ; n[1]=4.5-0.2j ; n[2]=1+0j
  ang=ang1*pi/180.

  for i in xrange(npt):
    temp[0]=thick
    Rtm[i],Rte[i],irad[i] = Rlayer(nlayer,n,temp,ang[i],freq)
    
  pl.plot(ang1, irad, 'b-',linewidth=1.0)
  pl.plot(ang1, abs(Rte),'g--', linewidth=1.0)
  pl.plot(ang1, abs(Rtm), 'g-.',linewidth=1.0)
  pl.legend( ('Irradiance','TE Refl. Coef','TM Refl. Coef'), loc='upper left')
  pl.xlim(0.0,90.0)
  pl.ylim(0.0,1.2)
  pl.xlabel('Incident Angle (deg)')
  pl.ylabel('Magnitude')
  # pl.title('8 THz; 25 deg.; n,k=4.5,0.2 in air ')
  pl.grid(True)
  pl.show()
  
#=============================================================================================

def thick_ang_3D():
    
  freq=8.0 #THz
  nlayer=1
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  n[0]=1.0+0j ; n[1]=4.5-0.2j ; n[2]=1+0j  # single layer in air 

  nthick=200 ; begthick=0.001 ; endthick=0.08
  nang=91 ; begang=0 ; endang=90
  thick=linspace(begthick,endthick,nthick) ; ang1=linspace(begang,endang,nang)
  temp=zeros(nlayer)
  ang1,thick=meshgrid(ang1,thick)
  ang=zeros(ang1.shape)
  ang=ang1*pi/180.0
  irad=zeros(thick.shape) ; Rtm=zeros(thick.shape,complex)
  Rte=zeros(thick.shape,complex)

  for i in xrange(thick.shape[0]):
    for j in xrange(ang.shape[1]):
      temp[0]=thick[i,j]
      Rtm[i,j],Rte[i,j],irad[i,j] = Rlayer(nlayer,n,temp,ang[i,j],freq)
      
  fig1=plt.figure()
  ax1=Axes3D(fig1)
  ax1.plot_surface(ang1,thick,abs(irad),rstride=1, cstride=1, cmap=cm.jet)
  ax1.set_xlabel('Incident Angle (deg)')
  ax1.set_ylabel('Layer Thickness (mm)')
  ax1.set_zlabel('Irradiance')
  fig2=plt.figure()
  ax2=Axes3D(fig2)
  ax2.plot_surface(ang1,thick,abs(Rtm),rstride=1, cstride=1, cmap=cm.jet)
  ax2.set_ylabel('Layer Thickness (mm)')
  ax2.set_xlabel('Incident Angle (deg)')
  ax2.set_zlabel('Rtm Magnitude')
  fig3=plt.figure()
  ax3=Axes3D(fig3)
  ax3.plot_surface(ang1,thick,abs(Rte),rstride=1, cstride=1, cmap=cm.jet)
  ax3.set_ylabel('Layer Thickness (mm)')
  ax3.set_xlabel('Incident Angle (deg)')
  ax3.set_zlabel('Rte Magnitude')
  plt.show()
  
#=============================================================================================

def thick_freq_3D():
    
  nlayer=1 ; ang1=25. # deg
  ang=ang1*pi/180.
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  n[0]=1.0+0j ; n[1]=4.5-0.2j ; n[2]=1+0j  # single layer in air 

  thick=linspace(0.0,0.05,100) ; freq=linspace(0.0,15.0,200) ; temp=zeros(nlayer)
  freq,thick=meshgrid(freq,thick)
  irad=zeros(thick.shape)

  for i in xrange(thick.shape[0]):
    for j in xrange(freq.shape[1]):
      temp[0]=thick[i,j]
      Rtm,Rte,irad[i,j] = Rlayer(nlayer,n,temp,ang,freq[i,j])
  
  fig1=plt.figure()
  ax1=Axes3D(fig1)
  ax1.plot_surface(thick,freq,irad,rstride=1, cstride=1, cmap=cm.jet)
  ax1.set_xlabel('Layer Thickness (mm)')
  ax1.set_ylabel('Frequency (THz)')
  ax1.set_zlabel('Irradiance')
  plt.show()

#=============================================================================================

def freq_ang_3D():
    
  nlayer=1 ; thick=0.01 #mm
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  n[0]=1.0+0j ; n[1]=4.5-0.2j ; n[2]=1+0j  # single layer in air 

  ang1=linspace(0.0,90.0,91) ; freq=linspace(0.0,15.0,200) ; temp=zeros(nlayer)
  freq,ang1=meshgrid(freq,ang1)
  ang=ang1*pi/180.
  irad=zeros(ang1.shape)

  for i in xrange(ang.shape[0]):
    for j in xrange(freq.shape[1]):
      temp[0]=thick
      Rtm,Rte,irad[i,j] = Rlayer(nlayer,n,temp,ang[i,j],freq[i,j])
      
  fig1=plt.figure()
  ax1=Axes3D(fig1)
  ax1.plot_surface(freq,ang1,irad,rstride=1, cstride=1, cmap=cm.jet)
  ax1.set_xlabel('Frequency (THz)')
  ax1.set_ylabel('Incident Angle (deg)')
  ax1.set_zlabel('Irradiance')
  plt.show()

#=============================================================================================

def n_k_3D():
    
  nn=40 ; begn=0.1 ; endn=5.0
  nk=50 ; begk=0.1 ; endk=1.5
  thick=0.05 #mm
  ang1=25. # deg
  freq=8.0 #THz
  ang=ang1*pi/180.0
  
  nlayer=1
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  n[0]=1.0+0.0j ; n[2]=1.0+0.0j # single layer in air 

  n_n=linspace(begn,endn,nn) ; n_k=linspace(begk,endk,nk)
  temp=zeros(nlayer)
  n_k,n_n=meshgrid(n_k,n_n)
  irad=zeros(n_k.shape)

  for i in xrange(n_n.shape[0]):
    for j in xrange(n_k.shape[1]):
      n.real[1]=n_n[i,j] ; n.imag[1]=-n_k[i,j]
      temp[0]=thick
      Rtm,Rte,irad[i,j] = Rlayer(nlayer,n,temp,ang,freq)
  
  fig1=plt.figure()
  ax1=Axes3D(fig1)
  ax1.plot_surface(n_n,n_k,irad,rstride=1, cstride=1, cmap=cm.jet)
  ax1.set_xlabel('Refractive Index, n')
  ax1.set_ylabel('Extinction Coeff., k')
  ax1.set_zlabel('Irradiance')
  plt.show()

#=============================================================================================

def n_k_thick_3D():
    
# exhausted search of "matched thickness" on n&k vs other variables

  nthick=3000 ; begthick=0.001 ; endthick=3.0
  nn=40 ; begn=0.1 ; endn=5.0
  nk=50 ; begk=0.1 ; endk=1.5
  ang1=25. # deg
  ang=ang1*pi/180.0
  
  freq=8.0 #THz
  if 1.0>=freq>0.5:
    endthick=2.0
  elif 1.5>=freq>1.0:
    endthick=1.0
  elif 2.0>=freq>1.5:
    endthick=0.8
  elif 9.0>=freq>2.0:
    endthick=0.2
  elif 16.0>=freq>8.0:
    endthick=0.1    
  nlayer=1
  n=zeros(nlayer+2,complex) # nlayer+2 media for n
  n[0]=1.0+0j ; n[1]=4.5-0.2j ; n[2]=1+0j  # single layer in air 

  thick=linspace(begthick,endthick,nthick) ; delthick=(endthick-begthick)/(nthick-1)
  n_n=linspace(begn,endn,nn) ; n_k=linspace(begk,endk,nk)
  temp=zeros(nlayer)
  n_k,n_n=meshgrid(n_k,n_n)
  zthick=zeros(n_n.shape)

  for i in xrange(n_n.shape[0]):
    for j in xrange(n_k.shape[1]):
      n.real[1]=n_n[i,j] ; n.imag[1]=-n_k[i,j]
      Rtm0,Rte0,irad0=Rair_hs(ang,n[1])
        # backward search for the matched thickness at which layered model converges
        # within 1% of half-space model
      for k in xrange(nthick-1,0,-1):
        temp[0]=thick[k]
        Rtm,Rte,irad = Rlayer(nlayer,n,temp,ang,freq)
        if abs((irad-irad0)/irad0)>0.01:
          zthick[i,j]=k*delthick+begthick
          print i,j,n[1],zthick[i,j]
          break
  
  fig1=plt.figure()
  ax1=Axes3D(fig1)
  ax1.plot_surface(n_n,n_k,zthick,rstride=1, cstride=1, cmap=cm.jet)
  ax1.set_xlabel('Refractive Index, n')
  ax1.set_ylabel('Extinction Coeff., k')
  ax1.set_zlabel('Matched Thickness (mm)')
  plt.show()

#=============================================================================================

def Rlayer(nlayer,n,thick,ang,freq): 

# last update: 18JAN2010

# n, thick MUST enter as numpy array
# ang, freq MUST enter as scalar
# thick in mm; ang in rad; freq in THz

# temp fix TM mode at 90deg (at which costh=0)
# two layers tested OK

# incident wave comes in at interface 0
# indices for sigma_te,sigma_tm: the nlayer+1 interfaces
# indices for thick: the nlayer layers
# indices for n,costh,nt_te,nt_tm: the nlayer+2 media

# n turns others to numpy arrays too

  if abs(ang-1.570796)<0.0001:  # when ang=90deg
    Rte=1.0+0.0j ; Rtm=-Rte; irad=1.0
    return Rtm,Rte,irad
  eye=0+1j
  FSWavNum=2.*pi*freq/0.299792458 # light speed in mm/ps
  tmp=n[0]*sin(ang) ; tmp=tmp*tmp 
  tmp2=1.0-tmp/(n*n)
  tmp3=sqrt(tmp2)
  costh=zeros(tmp2.shape[0],complex)
  for i in xrange(tmp2.shape[0]): 
    # the copmposite logics in following line can't be vectorized (not yet) 
    costh[i]=where(tmp2[i].real<0. and abs(tmp2[i].imag)<0.00001,-tmp3[i],tmp3[i])
  nt_te=n*costh ; nt_tm=n/costh                                #Orfanidis eq. (8.1.4)
  sigma_te=zeros(nlayer+2,complex) ; sigma_tm=zeros(nlayer+2,complex)
  # Orfanidis eq. (8.1.3); sigma_te[0] and signma_tm[0] not used
  sigma_te[1:nlayer+2]=(nt_te[0:nlayer+1]-nt_te[1:nlayer+2])/(nt_te[0:nlayer+1]+nt_te[1:nlayer+2])
  sigma_tm[1:nlayer+2]=(nt_tm[0:nlayer+1]-nt_tm[1:nlayer+2])/(nt_tm[0:nlayer+1]+nt_tm[1:nlayer+2])  
  
  Rte=sigma_te[nlayer+1] ; Rtm=sigma_tm[nlayer+1]
  for i in xrange(nlayer,0,-1):  # don't know how to vectorize this recursion yet
    tt=FSWavNum*n[i]*thick[i-1]*costh[i] # phase thickness 
    tt=exp(-2.0*eye*tt)
    Rte=(sigma_te[i]+Rte*tt)/(1.0+sigma_te[i]*Rte*tt)  # Orfanidis eq. (8.1.7)
    Rtm=(sigma_tm[i]+Rtm*tt)/(1.0+sigma_tm[i]*Rtm*tt)  
  irad=0.5*(abs(Rtm)**2+abs(Rte)**2) 
  
  return Rtm,Rte,irad 

#=============================================================================================

def Rair_hs(ang,nd):
  e=sin(ang)**2 ; c=cos(ang)
  d=nd*nd ; b=d*c ; a=sqrt(d-e)
  Rtm0=(a-b)/(a+b) ; Rte0=(c-a)/(c+a)
  irad0=0.5*(abs(Rtm0)**2+abs(Rte0)**2)
  return Rtm0,Rte0,irad0
    
main()

# print tmp3.real
# print tmp3.imag
# costh=where(real(tmp2)<0. and abs(imag(tmp2))<0.00001,-tmp3,tmp3)
# irad=0.5*(Rtm*Rtm.conj+Rte*Rte.conj)   does not work
