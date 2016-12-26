//
//  main.cpp
//  wavempi
//
//  Created by weixu on 11/2/16.
//  Copyright © 2016 weixu. All rights reserved.
//

#include <iostream>
#include <math.h>
#include "Timer.h"
#include <stdlib.h>   // atoi
#include "mpi.h"
#include <omp.h>

int default_size = 100;  // the default system size
int defaultCellWidth = 8;
double c = 1.0;      // wave speed
double dt = 0.1;     // time quantum
double dd = 2.0;     // change in system

using namespace std;

// create a simulation space
void init(double **z, int size){
    for(int k=0; k<3; k++)
        for ( int i = 0; i < size; i++ )
            for ( int j = 0; j < size; j++ )
                z[k][i*size+j] = 0.0; // no wave
}

void print(double **z, int size, int t){
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            cout<<z[t][i*size+j]<< " ";
        }
        cout<<endl;
    }
}

// time = 0;
// initialize the simulation space: calculate z[0][][]
void start_t0(int weight, double **z, int size){
    for( int i = 0; i < size; i++ ) {
        for( int j = 0; j < size; j++ ) {
            if( i > 40 * weight && i < 60 * weight  &&
               j > 40 * weight && j < 60 * weight ) {
                z[0][i*size+j] = 20.0;
            } else {
                z[0][i*size+j] = 0.0;
            }
        }
    }
}

void start_t1(double **z, int size){//??boundary equals to 0
    for(int j=0; j<size; j++){//when i equals to 0
        z[1][j]=0;
        z[1][(size-1)*size]=0;
        z[2][j]=0;
        z[2][(size-1)*size]=0;
    }
    for(int i=0; i<size; i++){
        z[1][i*size]=0;
        z[1][i*size+size-1]=0;
        z[2][i*size]=0;
        z[2][i*size+size-1]=0;
    }
    for(int i=1;i<size-1;i++){
        for(int j=1; j<size-1; j++){
            z[1][i*size+j]=z[0][i*size+j]+c*c/2.0*pow((double)(dt/dd),2.0)*(z[0][(i+1)*size+j]+z[0][(i-1)*size+j]+z[0][i*size+j+1]+z[0][i*size+j-1]-4.0*z[0][i*size+j]);
        }
    }
}

void calct2(double **z, int size, int t, int t_1, int t_2, int i, int j){
    z[t][i*size+j]=2.0*z[t_1][i*size+j]-z[t_2][i*size+j]+c*c*pow((double)(dt/dd),2.0)*(z[t_1][(i+1)*size+j]+z[t_1][(i-1)*size+j]+z[t_1][i*size+j+1]+z[t_1][i*size+j-1]-4.0*z[t_1][i*size+j]);
}

// simulate wave diffusion from time = 2
void start_t2(double **z, int size, int rank, int stripe, int t, int t_1, int t_2, int mpi_size) {
    if(rank==0){
        #pragma omp parallel for
        for(int i=1;i<stripe; i++){
            for(int j=1; j<size-1; j++){
               calct2(z,size,t,t_1,t_2,i,j);
            }
        }
    }else if(rank==mpi_size-1){
        #pragma omp parallel for
        for(int i=rank*stripe; i<size-1; i++){
            for(int j=1; j<size-1; j++){
                calct2(z,size,t,t_1,t_2,i,j);
            }
        }
    }else {
        #pragma omp parallel for
        for(int i=rank*stripe;i<(rank+1)*stripe;i++){
            for(int j=1;j<size-1;j++){
                calct2(z,size,t,t_1,t_2,i,j);
            }
        }
    } // end of simulation
}

int main( int argc, char *argv[] ) {
    int my_rank=0;            // used by MPI
    int mpi_size=1;           // used by MPI
    
    // verify arguments
    if ( argc != 5 ) {
        cerr << "usage: Wave2D size max_time interval" << endl;
        return -1;
    }
    int size = atoi( argv[1] );
    int max_time = atoi( argv[2] );
    int interval  = atoi( argv[3] );
    int nThreads = atoi(argv[4]);
    
    if ( size < 100 || max_time < 3 || interval < 0 ) {
        cerr << "usage: Wave2D size max_time interval" << endl;
        cerr << "       where size >= 100 && time >= 3 && interval >= 0" << endl;
        return -1;
    }
    int weight = size / default_size;
    //start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
    MPI_Status status;
   
    double **z = new double *[3];
    for(int i=0; i<3; i++){
        z[i]=new double[size*size];
    }
    // change # of threads
    omp_set_num_threads( nThreads );
    
    // start a timer
    Timer time;
    //all ranks initialize a space
    init(z,size);
    if(my_rank==0){
        time.start( );
    }

    /*MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&max_time,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&interval,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&weight,1,MPI_INT,0,MPI_COMM_WORLD);*/
    
    int stripe=size/mpi_size;

    start_t0(weight, z, size);
    start_t1(z, size);
    if(interval==1 && my_rank==0){
        cout<<interval<<endl;
        print(z, size, 1);
    }


    // simulate wave diffusion from time = 2
    int t_1=1;
    int t_2=0;
    if(my_rank == 0 && mpi_size == 1){
        for (int t = 2, step=2; step < max_time; t=(t+1)%3,t_1=(t_1+1)%3,t_2=(t_2+1)%3,step++){
            #pragma omp parallel for
            for(int i=1; i<size-1; i++){
                for(int j=1; j<size-1; j++){
                    calct2(z,size,t,t_1,t_2,i,j);
                }
            }
            if(interval!=0&&step%interval==0){
                cout<<interval<<endl;
                print(z, size, t);
            }
        }
    }else{
        for (int t = 2, step=2; step < max_time; t=(t+1)%3,t_1=(t_1+1)%3,t_2=(t_2+1)%3,step++) {
            start_t2(z, size, my_rank, stripe, t, t_1, t_2, mpi_size);
            if(my_rank==0){
                MPI_Send(z[t]+(stripe-1)*size,size,MPI_DOUBLE,1,0,MPI_COMM_WORLD);//right side
                MPI_Recv(z[t]+stripe*size,size,MPI_DOUBLE,1,0,MPI_COMM_WORLD,&status);
            }else if(my_rank%2==0){
                MPI_Send(z[t]+my_rank*stripe*size,size,MPI_DOUBLE,my_rank-1,0,MPI_COMM_WORLD);//left side
                if(my_rank!=mpi_size-1){
                    MPI_Send(z[t]+((my_rank+1)*stripe-1)*size,size,MPI_DOUBLE,my_rank+1,0,MPI_COMM_WORLD);//right side
                }
                MPI_Recv(z[t]+(my_rank*stripe-1)*size,size,MPI_DOUBLE,my_rank-1,0,MPI_COMM_WORLD,&status);//left
                if(my_rank!=mpi_size-1){
                    MPI_Recv(z[t]+(my_rank+1)*stripe*size,size,MPI_DOUBLE,my_rank+1,0,MPI_COMM_WORLD,&status);//right
                }
            }else{
                MPI_Recv(z[t]+(my_rank*stripe-1)*size,size,MPI_DOUBLE,my_rank-1,0,MPI_COMM_WORLD,&status);//left
                if(my_rank!=mpi_size-1){
                    MPI_Recv(z[t]+(my_rank+1)*stripe*size,size,MPI_DOUBLE,my_rank+1,0,MPI_COMM_WORLD,&status);//right
                }
                MPI_Send(z[t]+my_rank*stripe*size,size,MPI_DOUBLE,my_rank-1,0,MPI_COMM_WORLD);//left
                if(my_rank!=mpi_size-1){
                    MPI_Send(z[t]+((my_rank+1)*stripe-1)*size,size,MPI_DOUBLE,my_rank+1,0,MPI_COMM_WORLD);//right
                }
            }
            if(interval!=0&&step%interval==0){
                if(my_rank==0){
                    for(int rank=1; rank<mpi_size; rank++){
                        MPI_Recv(z[t]+stripe*size*rank,stripe*size,MPI_DOUBLE,rank,0,MPI_COMM_WORLD,&status);
                    }
                    cout<<step<<endl;
                    print(z, size, t);
                }else {
                    MPI_Send(z[t]+stripe*size*my_rank,stripe*size,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
                }
            }
        } // end of simulation
    }

    
    // finish the timer
    if(my_rank==0){
        cerr << "Elapsed time = " << time.lap( ) << endl;
    }
    for(int i=0; i<3; i++){
        delete[] z[i];
    }
    delete[] z;
    MPI_Finalize();
    return 0;
}
