# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:23:23 2016

@author: hunterdavis
"""


import numpy

lattice=numpy.array([[1,0,0], [0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],\
[-1,-1,-1],\
[-1,-1,1],\
[-2,-2,-1],\
[-2,-2,1],\
[-2,-1,-2],\
[-2,-1,0],\
[-2,-1,2],\
[-2,0,-1],\
[-2,0,1],\
[-2,1,-2],\
[-2,1,0],\
[-2,1,2],\
[-2,2,-1],\
[-2,2,1],\
[-1,-2,-2],\
[-1,-2,0],\
[-1,-2,2],\
[-1,0,-2],\
[-1,0,2],\
[-1,2,-2],\
[-1,2,0],\
[-1,2,2],\
[0,-2,-1],\
[0,-2,1],\
[0,-1,-2],\
[0,-1,2],\
[0,1,-2],\
[0,1,2],\
[0,2,-1],\
[0,2,1],\
[1,-2,-2],\
[1,-2,0],\
[1,-2,2],\
[1,0,-2],\
[1,0,2],\
[1,2,-2],\
[1,2,0],\
[1,2,2],\
[2,-2,-1],\
[2,-2,1],\
[2,-1,-2],\
[2,-1,0],\
[2,-1,2],\
[2,0,-1],\
[2,0,1],\
[2,1,-2],\
[2,1,0],\
[2,1,2],\
[2,2,-1],\
[2,2,1],\
[-2,-2,3],\
[-2,-2,-3],\
[-2,0,3],\
[-2,0,-3],\
[-2,2,3],\
[-2,2,-3],\
[-2,3,-2],\
[-2,-3,-2],\
[-2,3,0],\
[-2,-3,0],\
[-2,3,2],\
[-2,-3,2],\
[-1,-1,3],\
[-1,-1,-3],\
[-1,1,3],\
[-1,1,-3],\
[-1,3,-1],\
[-1,-3,-1],\
[-1,3,1],\
[-1,-3,1],\
[0,-2,3],\
[0,-2,-3],\
[0,0,3],\
[0,0,-3],\
[0,2,3],\
[0,2,-3],\
[0,3,-2],\
[0,-3,-2],\
[0,3,0],\
[0,-3,0],\
[0,3,2],\
[0,-3,2],\
[1,-1,3],\
[1,-1,-3],\
[1,1,3],\
[1,1,-3],\
[1,3,-1],\
[1,-3,-1],\
[1,3,1],\
[1,-3,1],\
[2,-2,3],\
[2,-2,-3],\
[2,0,3],\
[2,0,-3],\
[2,2,3],\
[2,2,-3],\
[2,3,-2],\
[2,-3,-2],\
[2,3,0],\
[2,-3,0],\
[2,3,2],\
[2,-3,2],\
[3,-2,-2],\
[-3,-2,-2],\
[3,-2,0],\
[-3,-2,0],\
[3,-2,2],\
[-3,-2,2],\
[3,-1,-1],\
[-3,-1,-1],\
[3,-1,1],\
[-3,-1,1],\
[3,0,-2],\
[-3,0,-2],\
[3,0,0],\
[-3,0,0],\
[3,0,2],\
[-3,0,2],\
[3,1,-1],\
[-3,1,-1],\
[3,1,1],\
[-3,1,1],\
[3,2,-2],\
[-3,2,-2],\
[3,2,0],\
[-3,2,0],\
[3,2,2],\
[-3,2,2],\
[1,-3,-3],\
[1,-3,3],\
[1,3,-3],\
[1,3,3],\
[-1,-3,-3],\
[-1,-3,3],\
[-1,3,-3],\
[-1,3,3],\
[3,-1,-3],\
[3,-1,3],\
[-3,-1,-3],\
[-3,-1,3],\
[3,1,-3],\
[3,1,3],\
[-3,1,-3],\
[-3,1,3],\
[3,-3,-1],\
[3,3,-1],\
[-3,-3,-1],\
[-3,3,-1],\
[3,-3,1],\
[3,3,1],\
[-3,-3,1],\
[-3,3,1],\
[-3,-3,-3],\
[-3,-3,3],\
[-3,3,-3],\
[-3,3,3],\
[3,-3,-3],\
[3,-3,3],\
[3,3,-3],\
[3,3,3]])

def OnAxisDipole(x,y,z,M):
    B=M*10**(-7)/((x**2+y**2+z**2)**(3/2))*(3*z**2/(x**2+y**2+z**2)-1);
    return B
radius=5; #Cell Radius in um

MPos=numpy.ones([1,3])
MMomentStore=numpy.array([])

centers=numpy.sqrt(2)*radius*lattice;

t=10000; #Total Time steps
tau=.001; #Time step size in ms
dd=14; #Leave Alone (artifact from aqp code)
taucp=5.5; ##Carr-Purcell Parameter in ms
gamma=42.5781*10**6; ##Gyromagnetic Ratio in inverse seconds

box_size=2*numpy.sqrt(2)*radius;


#throw out dipoles that fall outside of box bounds
numwater=800; #number of water molecules to sample


T_2Flag=1;
Magstore=numpy.zeros([numwater,t]);
Bzstore=numpy.zeros([numwater,t]);

#Deal with partial cells by throwing out placed NP outside of box boundary

numdipoles=numpy.shape(MMoment)[0];
def T2relaxation(xx):

    numpy.random.seed(xx)

    flipflag=1;
    water_xyz=numpy.zeros([t,3]);
    water_xyz[0,:]=box_size*(numpy.random.rand(1,3)-.5);
    xcount=numpy.zeros([t,1]);
    P_Expressing=.012; ##Permeability in micron per ms#
    D_Cellular=1.0; ###D in micron^2 per ms#
    D_ExtraCellular=2.34; ###D in micron^2 per ms#
    aqp_Reflec=1-(tau/(6*D_Cellular))**(.5)*4*P_Expressing;
    R_io=aqp_Reflec;
    R_oi=1-((1-R_io)*numpy.sqrt(D_Cellular/D_ExtraCellular));
    Phase=numpy.zeros([t,1]);
    Mag=numpy.ones([t,1]);
    Bzv=numpy.zeros([t,1]);
    while numpy.min(numpy.squeeze(numpy.sum((water_xyz[0,:]-dipole_xyz)**2,axis=1)-pradius**2))<0:
        water_xyz[0,:]=3*box_size*(numpy.random.rand(1,3)-.5);

    for j in list(numpy.array(range(t-1))+1):
        q=numpy.argmin(numpy.sum((water_xyz[j-1,:]-centers)**2,axis=1))
        bk=centers[q,:];  
        Bz=0
        Xr=numpy.random.rand();     
        if numpy.sum((water_xyz[j-1,:]-bk)**2)<radius**2:
            water_xyz[j,:]=water_xyz[j-1,:]+numpy.random.randn(1,3)*numpy.sqrt(numpy.pi/2)*numpy.sqrt(2*D_Cellular*tau);
        else:
            water_xyz[j,:]=numpy.squeeze(water_xyz[j-1,:])+numpy.random.randn(1,3)*numpy.sqrt(numpy.pi/2)*numpy.sqrt(2*D_ExtraCellular*tau);            
        
        if (numpy.sum((water_xyz[j-1,:]-bk)**2)<radius**2 and numpy.sum((water_xyz[j,:]**2-bk))>radius**2 and Xr<R_io) or (numpy.sum((water_xyz[j-1,:]**2-bk))>radius**2 and numpy.sum((water_xyz[j,:]**2-bk))<(radius)**2 and Xr<R_oi):
            water_xyz[j,:]=water_xyz[j-1,:];   
        

        xcount[j]=xcount[j-1];
        if water_xyz[j,0]>3*box_size/2:
           water_xyz[j,0]=water_xyz[j,1]-box_size;
           xcount[j]=xcount[j]+1;
        
        
        if water_xyz[j,0]<-3*box_size/2:
           water_xyz[j,0]=water_xyz[j,1]+box_size;
           xcount[j]=xcount[j]-1;
        
        
        if water_xyz[j,1]>3*box_size/2 or water_xyz[j,2]<-3*box_size/2:
           water_xyz[j,1]=water_xyz[j,1]-numpy.sign(water_xyz[j,1])*box_size;
        
        
        if water_xyz[j,2]>3*box_size/2 or water_xyz[j,2]<-3*box_size/2:
           water_xyz[j,2]=water_xyz[j,2]-numpy.sign(water_xyz[j,2])*box_size; 
        
        if numpy.min(numpy.squeeze(numpy.sum((water_xyz[j,:]-dipole_xyz)**2,axis=1))-pradius**2) < 0:
           water_xyz[j,:]=water_xyz[j-1,:];
        
        for xx in range(numdipoles):
           Bz=Bz+OnAxisDipole((water_xyz[j,0]-dipole_xyz[xx,0])*10**(-6),(water_xyz[j,1]-dipole_xyz[xx,1])*10**(-6),(water_xyz[j,2]-dipole_xyz[xx,2])*10**(-6),MMoment[xx]);
        
        Phase[j]=Phase[j-1]+gamma*2*numpy.pi*tau*Bz; #2pi term converts to phase velocity            
        if j/(taucp/tau)==numpy.floor(j/(taucp/tau)) and not ((j/(2*taucp/tau))==numpy.floor(j/(2*taucp/tau))) and T_2Flag==1:
           Phase[j]=-Phase[j];
           flipflag=flipflag+1
        
        Mag[j]=numpy.cos(Phase[j]);
        Bzv[j]=Bz;
    
    return Mag
    
from multiprocessing import Pool
pool = Pool(4)
Magstore=pool.map(T2relaxation,numpy.random.randint(1,high=100000000,size=numwater))
Magstore=numpy.array(Magstore)

import matplotlib.pyplot as plt
Sig=numpy.abs(numpy.mean(Magstore,axis=0));


plt.plot(Sig)

#numpy.savetxt('NPClustered'+str(numpy.random.randint(10000))+'.out', Magstore, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
