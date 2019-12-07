#include <iostream>
#include "double.h"
#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>
#include "mkl.h"
#define N 10001
double matA[N*N],matB[N*N],matCm[N*N],matCm2[N*N];
int main()
{
    MPI_Init(0,NULL);
    int rs,rank,size,slice,K;
    int n = 1;
    double *matC=matCm,*matC2=matCm2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    rs=slice=N/size;
    K=rank*slice*N+N;
    if(rank==0)
    {
        rs++;K=0;
        input(matA, matB);
        memcpy(matC, matA, sizeof(double[N * N]));
            MPI_Bcast(matC,N*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Bcast(matB,N*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Bcast(matA,N*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    for (int k = 0; k < n; ++k)
    {
        #pragma omp parallel for
        for (int i = 0; i < N * N; ++i)
            matA[i] += matB[i];
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rs,N,N,1.0,matC,N,matA,N,0.0,matC2,N);
        double *t = matC;
        matC = matC2;
        matC2 = t;
    }
    MPI_Gather(matC+K,slice*N,MPI_DOUBLE,matC+N,slice*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Finalize();
    if(rank==0)
    {
        output(matC, n);
    }
    return 0;
}
