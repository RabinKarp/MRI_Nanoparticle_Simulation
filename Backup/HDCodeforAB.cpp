//
//  HDCodeforAB.cpp
//  
//
//  Created by Hunter Davis on 7/13/16.
//
//
//
//  main.cpp
//  RandomNumberTesting
//
//  Created by Hunter Davis on 7/11/16.
//  Copyright (c) 2016 HunterDavis. All rights reserved.
//

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <random>
#include <fstream>
#include <thread>
#include <vector>

#define NUM_THREADS     8

using namespace std;

const double DCellular=.5547;
const double DExtraCellular=1.6642;
const double CellPermeability=.039;

const double tau=1e-6; /*Time step in ms*/
const int totaltime=1e7; /*total time steps*/
const double cellradius=10;
const double boxsize=6*sqrt(2)*cellradius; /*boxsize is defined as one full leg of the entire boundary box*/
const int numnanoparticles=100;
const double particler=1;
const double Mmoment=4e-14;
const double gammaw=42.7e6;
const double pi=3.14159265;
const int numwater=1e2;
const double rio=1-sqrt((tau/(6*DCellular)))*4*CellPermeability;
const double roi=1-((1-rio)*sqrt(DCellular/DExtraCellular));

const double taucp=4;
const int tcpn=taucp/tau;

/*I use the array fccseed as the unit lattice and multiply by cell size to generate the actual array*/

double fccseed[172][3] = {{1,0,0},{0,1,0},{0,0,1},{-1,0,0},{0,-1,0},\
    {0,0,-1},{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},{-1,1,1},{-1,1,-1},\
    {-1,-1,-1},{-1,-1,1},{-2,-2,-1},{-2,-2,1},{-2,-1,-2},{-2,-1,0},\
    {-2,-1,2},{-2,0,-1},{-2,0,1},{-2,1,-2},{-2,1,0},{-2,1,2},{-2,2,-1},\
    {-2,2,1},{-1,-2,-2},{-1,-2,0},{-1,-2,2},{-1,0,-2},{-1,0,2},{-1,2,-2},\
    {-1,2,0},{-1,2,2},{0,-2,-1},{0,-2,1},{0,-1,-2},{0,-1,2},{0,1,-2},\
    {0,1,2},{0,2,-1},{0,2,1},{1,-2,-2},{1,-2,0},{1,-2,2},{1,0,-2},{1,0,2},\
    {1,2,-2},{1,2,0},{1,2,2},{2,-2,-1},{2,-2,1},{2,-1,-2},{2,-1,0},{2,-1,2},\
    {2,0,-1},{2,0,1},{2,1,-2},{2,1,0},{2,1,2},{2,2,-1},{2,2,1},{-2,-2,3},\
    {-2,-2,-3},{-2,0,3},{-2,0,-3},{-2,2,3},{-2,2,-3},{-2,3,-2},{-2,-3,-2},\
    {-2,3,0},{-2,-3,0},{-2,3,2},{-2,-3,2},{-1,-1,3},{-1,-1,-3},{-1,1,3},\
    {-1,1,-3},{-1,3,-1},{-1,-3,-1},{-1,3,1},{-1,-3,1},{0,-2,3},{0,-2,-3},\
    {0,0,3},{0,0,-3},{0,2,3},{0,2,-3},{0,3,-2},{0,-3,-2},{0,3,0},{0,-3,0},\
    {0,3,2},{0,-3,2},{1,-1,3},{1,-1,-3},{1,1,3},{1,1,-3},{1,3,-1},{1,-3,-1},\
    {1,3,1},{1,-3,1},{2,-2,3},{2,-2,-3},{2,0,3},{2,0,-3},{2,2,3},{2,2,-3},\
    {2,3,-2},{2,-3,-2},{2,3,0},{2,-3,0},{2,3,2},{2,-3,2},{3,-2,-2},\
    {-3,-2,-2},{3,-2,0},{-3,-2,0},{3,-2,2},{-3,-2,2},{3,-1,-1},{-3,-1,-1},\
    {3,-1,1},{-3,-1,1},{3,0,-2},{-3,0,-2},{3,0,0},{-3,0,0},{3,0,2},{-3,0,2},\
    {3,1,-1},{-3,1,-1},{3,1,1},{-3,1,1},{3,2,-2},{-3,2,-2},{3,2,0},{-3,2,0},\
    {3,2,2},{-3,2,2},{1,-3,-3},{1,-3,3},{1,3,-3},{1,3,3},{-1,-3,-3},\
    {-1,-3,3},{-1,3,-3},{-1,3,3},{3,-1,-3},{3,-1,3},{-3,-1,-3},{-3,-1,3},\
    {3,1,-3},{3,1,3},{-3,1,-3},{-3,1,3},{3,-3,-1},{3,3,-1},{-3,-3,-1},\
    {-3,3,-1},{3,-3,1},{3,3,1},{-3,-3,1},{-3,3,1},{-3,-3,-3},{-3,-3,3},\
    {-3,3,-3},{-3,3,3},{3,-3,-3},{3,-3,3},{3,3,-3},{3,3,3}};

/*FCC will eventually be filled as the array of coordinates of cell centers*/
double fcc[172][3];


/*Bfield calculated in meters. In order to accurately calculate here, I give the function displacements*1e-6 to calculate the local B field*/

double Bfield(double x, double y, double z,double M)
{
    double B;
    B=M*1e-7/(pow(x*x+y*y+z*z,1.5))*(3*z*z/(x*x+y*y+z*z)-1);
    return B;
};

/*generates a random number between one and 1e6 divide by 1.0e6 to get a double bewteen 0 and 1 (labeled plac)*/
double randinit()
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<> dis(1,1e6);
    double plac;
    plac=dis(gen)/1.0e6;
    return plac;
}

/*double to store particle position*/
double particlexyz[numnanoparticles][3];

/*random seed*/
std::mt19937_64 generator (time(NULL));

/*Generates a random displacement for a single dimension given the argument for the local diffusion constant D (different than AB code, which generates the size of a 3D displacement)*/

double Diffuse(double D)
{
    
    std::normal_distribution<double> distribution(0.0,1.0);
    double r;
    r=distribution(generator) * sqrt(pi*D*tau);
    return r;
};

/*Mag stores the water magnetization at each timepoint for each thread. Magstore is the sum across threads*/
double Mag[totaltime][NUM_THREADS];
double Magstore[totaltime];

void wsim(int threadid)
{
    int tid = threadid;
    cout<<threadid;
    
    /*Main Loop Below*/
    /*for loop is water(time()) nested, so we will deal with one water molecule's entire path first.*/
    for (int nw=0; nw<numwater; nw++) {
        
        double dx;
        double dy;
        double dz;
        double xt;
        double yt;
        double zt;
        double dxf;
        double dyf;
        double dzf;
        double dxt;
        double dyt;
        double dzt;
        double rsqf;
        double rsqt;
        double tempdxt;
        double tempdyt;
        double tempdzt;
        double temprsqt;
        double temprsqf;
        
        /*Initialize x,y, and z coordinates for the molecule inside the inner boxas xf, yf, zf.*/
        double xf=(randinit()-.5)*boxsize/3;
        double yf=(randinit()-.5)*boxsize/3;
        double zf=(randinit()-.5)*boxsize/3;
        
        /*Initialize cell 0 as the resident cell and initialize phase and Bzfield to 0*/
        int rescellf=0;
        int rescellt=0;
        double xcount=0;
        double phase=0;
        double Bzfield=0;
        double flipflag=0;
        
        for (int npn=0; npn < numnanoparticles; npn++)
        {
            if ((pow((particlexyz[npn][0]-xf),2)+pow((particlexyz[npn][1]-yf),2)+pow((particlexyz[npn][2]-zf),2))<(particler*particler))
            {
                xf=(randinit()-.5)*boxsize/3;
                yf=(randinit()-.5)*boxsize/3;
                zf=(randinit()-.5)*boxsize/3;
            }
        }
        
        dxf=xf-fcc[rescellf][0];
        dyf=yf-fcc[rescellf][1];
        dzf=zf-fcc[rescellf][2];
        
        rsqf=dxf*dxf+dyf*dyf+dzf*dzf;/*calculate and set square displacement from cell 0*/
        
        for (int cns=1; cns<172; cns++) /*Find Cell closest to initialized water position*/
        {
            /*Loop through all cells and find the cell that the water molecule is dropped closest to by calculating temprsqf and comparing it t rsqf.*/
            dxf=xf-fcc[cns][0];
            dyf=yf-fcc[cns][1];
            dzf=zf-fcc[cns][2];
            temprsqf=dxf*dxf+dyf*dyf+dzf*dzf;
            
            if (temprsqf<rsqf)
            {
                rsqf=temprsqf;
                rescellf=cns;
            }
        }
        
        /*Time Loop starts here*/
        
        for (int t=1; t<totaltime;t++) {
            
            /*from resident cell set at end of previous loop*/
            dxf=xf-fcc[rescellf][0];
            dyf=yf-fcc[rescellf][1];
            dzf=zf-fcc[rescellf][2];
            
            rsqf=dxf*dxf+dyf*dyf+dzf*dzf;
            
            if (rsqf<cellradius*cellradius) {
                xt=xf+Diffuse(DCellular);
                yt=yf+Diffuse(DCellular);
                zt=zf+Diffuse(DCellular);
            }
            else{
                xt=xf+Diffuse(DExtraCellular);
                yt=yf+Diffuse(DExtraCellular);
                zt=zf+Diffuse(DExtraCellular);
            }
            
            dxt=xt-fcc[rescellf][0];
            dyt=yt-fcc[rescellf][1];
            dzt=zt-fcc[rescellf][2];
            
            rsqt=dxt*dxt+dyt*dyt+dzt*dzt;
            
            
            if (rsqt > cellradius*cellradius)
            {
                if (rsqf < (cellradius*cellradius) && randinit()<rio)
                {
                    xt=xf;
                    yt=yf;
                    zt=zf;
                    rescellt=rescellf;
                }
                else
                {
                    for (int cn=1; cn<173; cn++) /*Identify resident cell for the attempted water step*/
                    {
                        tempdxt=xt-fcc[cn][0];
                        tempdyt=yt-fcc[cn][1];
                        tempdzt=zt-fcc[cn][2];
                        temprsqt=tempdxt*tempdxt+tempdyt*tempdyt+tempdzt*tempdzt;
                        
                        if (temprsqt<rsqt)
                        {
                            rsqt=temprsqt;
                            rescellt=cn;
                        }
                    }
                    if (rsqt < (cellradius*cellradius) && randinit()<roi && std::max(xt*xt,yt*yt)<boxsize*boxsize/4)
                    {
                        xt=xf;
                        yt=yf;
                        zt=zf;
                        rescellt=rescellf;
                    }
                }
            }
            else if (rsqf > cellradius*cellradius && randinit()<roi)
            {
                xt=xf;
                yt=yf;
                zt=zf;
                rescellt=rescellf;
            }
            
            
            if (xt>boxsize/2)
            {
                xt-=boxsize;
                xcount+=1;
            }
            
            if (yt>boxsize/2)
            {
                yt-=boxsize;
            }
            
            if (zt>boxsize/2)
            {
                zt-=boxsize;
            }
            if (xt < -1*boxsize/2)
            {
                xt+=boxsize;
                xcount-=1;
            }
            
            if (yt < -1*boxsize/2)
            {
                yt+=boxsize;
            }
            
            if (zt < -1*boxsize/2)
            {
                zt+=boxsize;
            }
            
            for (int npn=0; npn<numnanoparticles; npn++)
            {
                dx=particlexyz[npn][0]-xt;
                dy=particlexyz[npn][1]-yt;
                dz=particlexyz[npn][2]-zt;
                if ((pow(dx,2)+pow(dy,2)+pow(dz,2))<(particler*particler))
                {
                    xt=xf;
                    yt=yf;
                    zt=zf;
                    dx=particlexyz[npn][0]-xf;
                    dy=particlexyz[npn][1]-yf;
                    dz=particlexyz[npn][2]-zf;
                }
                Bzfield+=Bfield(dx*1e-6,dy*1e-6,dz*1e-6,Mmoment);
            }
            
            
            phase+=Bzfield*2*pi*gammaw*1e-3*tau;
            
            if (t % (2*tcpn) == tcpn)
            {
                phase*=-1;
                flipflag+=1;
            }
            
            
            Bzfield=0;
            
            if (nw==0)
            {
                Mag[t][tid]=cos(phase);
            }
            else{
                Mag[t][tid]+=cos(phase);
            }
            
            xf=xt;
            yf=yt;
            zf=zt;
            rescellf=rescellt;
        }
        std::cout<< nw;
        std::cout<< "\n";
    }
}


int main(){
    /*initialize the fcc lattice*/
    for (int i=0; i<172; i++) {
        for (int j=0; j<3; j++) {
            fcc[i][j]=fccseed[i][j]*sqrt(2)*cellradius;
        }
    }
    
    
    /*Initialize nanoparticle positions randomly*/
    for (int npn=0; npn<numnanoparticles; npn++)
    {
        for (int dim=0; dim<3; dim++)
        {
            particlexyz[npn][dim]=(randinit()-.5)*boxsize;
        }
    }
    
    
    
    for (int ti=0; ti<NUM_THREADS; ti++) {
        Mag[0][ti]=numwater;
    }
    std::vector<std::thread> threads;
    
    for (int i=1; i<=NUM_THREADS; ++i)
        threads.push_back(std::thread(wsim,i-1));
    
    for (auto& th : threads) th.join();
    
    cout<<"Summing Data from threads";
    
    for (int j=1; j<totaltime; j++)
    {
        Magstore[j]=Mag[j][0];
        for (int i=1; i<NUM_THREADS; i++)
        {
            Magstore[j]+=Mag[j][i];
        }
    }
    Magstore[0]=numwater*NUM_THREADS;
    std::ofstream myfile;
    myfile.open ("trialoutput4highres4.txt");
    
    for (int q=0; q<totaltime; q++) {
        myfile << Magstore[q]<< endl;
        cout <<Magstore[q]<< endl;
    }
    myfile.close();
    
    
    return 0;
    
};
