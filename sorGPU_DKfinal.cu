/*
  Uses N blocks with N threads
  SOR Stokes Flow with no slip b.c. on top/bottom and no flux b.c. on left/right written by Dmitriy Kats

  Inputs: N is the number of grid points in each direction, 
  	  	  mu is the viscosity
  	  	  Pdiff is the pressure drop in the x direction
  	  	  omega is the SOR factor
  	  	  toltau is the tolerance of the residual

  Outputs: The final velocities and pressure

 */


#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include <time.h>


//Kernels to udpate u, v, and p
//The inputs also considers if it is a red or black point udpate
__global__ void update_u(double* U, double* Uresid, double* P, double* Presid, double* FAC1, double* OMEGA, int RedorBlack);
__global__ void update_v(double* V, double* Vresid, double* P, double* Presid, double* FAC1, double* OMEGA, int RedorBlack);
__global__ void update_p(double* U, double* V, double* P, double* Presid, double* FAC1, double* OMEGA, double* Pdiff, int RedorBlack);

__device__ static int dev_N;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}



int main (int argc, char * argv[]){

	// Choose the GPU card
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.multiProcessorCount = 13; 
	cudaChooseDevice(&dev, &prop);
	cudaSetDevice(dev);

	// Create the CUDA events that will be used for timing the kernel function
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Click, the timer has started running
	cudaEventRecord(start, 0);



	int N;
	double mu, pdiff, omega, toltau;

	N=atoi(argv[1]);
	mu=atof(argv[2]);
	pdiff=atof(argv[3]);
	omega=atof(argv[4]);
	toltau=atof(argv[5]);

	double dx=1.0/((double)N-1.0);
	double fac1=dx/mu; //precompute the factor

	double residABSMAX = 99.0;
	int numberOfIterations=0;
	
	double* dev_fac1;
	double* dev_omega;
	double* dev_pdiff;

	double *dev_u, *dev_uresid;
	double *dev_v, *dev_vresid;
	double *dev_p, *dev_presid;
	
	//allocate memory for the velocities and pressure

	double *u = (double*)malloc(N*(N-1)*sizeof(double));
	double *uresid = (double*)malloc(N*(N-1)*sizeof(double));

	double *v = (double*)malloc((N-1)*N*sizeof(double));
	double *vresid = (double*)malloc((N-1)*N*sizeof(double));

	double *p = (double*)malloc((N+1)*(N-1)*sizeof(double));
	double *presid = (double*)malloc((N+1)*(N-1)*sizeof(double));
	
	//allocate Cuda memory
	cudaMalloc((void**)&dev_fac1, sizeof(double));
	cudaMalloc((void**)&dev_omega, sizeof(double));
	cudaMalloc((void**)&dev_pdiff, sizeof(double));
	cudaMalloc((void**)&dev_u, N*(N-1)*sizeof(double));
	cudaMalloc((void**)&dev_uresid, N*(N-1)*sizeof(double));
	cudaMalloc((void**)&dev_v, (N-1)*N*sizeof(double));
	cudaMalloc((void**)&dev_vresid, (N-1)*N*sizeof(double));
	cudaMalloc((void**)&dev_p, (N+1)*(N-1)*sizeof(double));
	cudaMalloc((void**)&dev_presid, (N+1)*(N-1)*sizeof(double));

	//Intialize to zero
	int i, j;
	for(i=0; i<N; i++)
	{
		for(j=0; j<N-1; j++)
		{
			u[i+j*N]=0.0;  
			uresid[i+j*N]=0.0;
		}
	}

	for(i=0; i<N-1; i++)
	{
		for(j=0; j<N; j++)
		{
			v[i+j*(N-1)]=0.0;   
			vresid[i+j*(N-1)]=0.0;
		}
	}

	for(i=0; i<N+1; i++)
	{
		for(j=0; j<N-1; j++)
		{
			p[i+j*(N+1)]=0.0; 
			presid[i+j*(N+1)]=0.0;
		}
	}
	
	//Copy the values to the device
	cudaMemcpy(dev_u, u, N*(N-1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_uresid, uresid, N*(N-1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v, v, (N-1)*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vresid, vresid, (N-1)*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p, p, (N+1)*(N-1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_presid, presid, (N+1)*(N-1)*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(dev_N, &N, sizeof(int));
	cudaMemcpy(dev_fac1, &fac1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_omega, &omega, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pdiff, &pdiff, sizeof(double), cudaMemcpyHostToDevice);

	
	dim3 meshDim(N,N); //This one will be for the velocities
	dim3 meshDim2(N+1,N); //This one will be for the pressure


	while(residABSMAX>=toltau)
	{
		residABSMAX=0.1*toltau;
		//Solve in the next six lines
		update_u<<<meshDim,1>>>(dev_u, dev_uresid, dev_p, dev_presid, dev_fac1, dev_omega, 0);
		update_u<<<meshDim,1>>>(dev_u, dev_uresid, dev_p, dev_presid,  dev_fac1, dev_omega, 1);
		update_v<<<meshDim,1>>>(dev_v, dev_vresid, dev_p, dev_presid,  dev_fac1, dev_omega, 0);
		update_v<<<meshDim,1>>>(dev_v, dev_vresid, dev_p, dev_presid, dev_fac1, dev_omega, 1);
		update_p<<<meshDim2,1>>>(dev_u, dev_v, dev_p, dev_presid,  dev_fac1, dev_omega, dev_pdiff, 0);
		update_p<<<meshDim2,1>>>(dev_u, dev_v, dev_p, dev_presid,  dev_fac1, dev_omega, dev_pdiff, 1);
		
		
		//This is slow but I ran out of time
		//Copy the residuals to the host to find the max residual
		cudaMemcpy(uresid, dev_uresid, N*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(vresid, dev_vresid, (N-1)*N*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(presid, dev_presid, (N+1)*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);

		for(i=0; i<N; i++)
		{
			for(j=0; j<N-1; j++)
			{
				if(fabs(uresid[i+j*N])>residABSMAX)
				{
					residABSMAX=fabs(uresid[i+j*N]);
				}
			}
		}

		for(i=0; i<N-1; i++)
		{
			for(j=0; j<N; j++)
			{
				if(fabs(vresid[i+j*(N-1)])>residABSMAX)
				{
					residABSMAX=fabs(vresid[i+j*(N-1)]);
				}
			}
		}

		for(i=0; i<N+1; i++)
		{
			for(j=0; j<N-1; j++)
			{
				if(fabs(presid[i+j*(N+1)])>residABSMAX)
				{
					residABSMAX=fabs(presid[i+j*(N+1)]);
				}
			}
		}

		
		//Check for errors
		gpuErrchk(cudaPeekAtLastError() );
		gpuErrchk(cudaDeviceSynchronize() );

		numberOfIterations+=1;


		if (numberOfIterations>10000)
		{	//fail safe to save data and exit
			cudaMemcpy(u, dev_u, N*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(v, dev_v, (N-1)*N*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(p, dev_p, (N+1)*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);

			printf("Reached fail safe. The max residual is %10e. The number of iterations is %i\n", residABSMAX, numberOfIterations);
			FILE *fpu = fopen("StokesU.out", "wb");
			fwrite(u, sizeof(double), N*(N-1), fpu);
			fclose (fpu);
			FILE *fpv = fopen("StokesV.out", "wb");
			fwrite(v, sizeof(double), (N-1)*N, fpv);
			fclose (fpv);
			FILE *fpP = fopen("StokesP.out", "wb");
			fwrite(p, sizeof(double), (N+1)*(N-1), fpP);
			fclose (fpP);

			cudaFree(dev_u);
			cudaFree(dev_uresid);
			cudaFree(dev_v);
			cudaFree(dev_vresid);
			cudaFree(dev_p);
			cudaFree(dev_presid);
			cudaFree(dev_fac1);
			cudaFree(dev_omega);
			cudaFree(dev_pdiff);

			free(u);
			free(uresid);
			free(v);
			free(vresid);
			free(p);
			free(presid);

			return 0;
		}
	}

	cudaMemcpy(u, dev_u, N*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, dev_v, (N-1)*N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(p, dev_p, (N+1)*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);
	//export the data
	FILE *fpu = fopen("StokesU.out", "wb");
	fwrite(u, sizeof(double), N*(N-1), fpu);
	fclose (fpu);
	FILE *fpv = fopen("StokesV.out", "wb");
	fwrite(v, sizeof(double), (N-1)*N, fpv);
	fclose (fpv);
	FILE *fpP = fopen("StokesP.out", "wb");
	fwrite(p, sizeof(double), (N+1)*(N-1), fpP);
	fclose (fpP);

	//stop the timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// The elapsed time is computed by taking the difference between start and stop
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("N:%i omega:%f\n", N, omega);
	printf("The max residual is %10e and the number of iterations is %i\n", residABSMAX, numberOfIterations);
	printf("Time: %gms\n", elapsedTime);

	//clean up timer
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);


	cudaFree(dev_u);
	cudaFree(dev_uresid);
	cudaFree(dev_v);
	cudaFree(dev_vresid);
	cudaFree(dev_p);
	cudaFree(dev_presid);
	cudaFree(dev_fac1);
	cudaFree(dev_omega);

	free(u);
	free(uresid);
	free(v);
	free(vresid);
	free(p);
	free(presid);

	return 0;
}

__global__ void update_u(double* U, double* Uresid, double* P, double* Presid, double* FAC1, double* OMEGA, int RorB)
{
	int EvenOrOdd=(blockIdx.x+blockIdx.y)%2;

	int u_ij00 = blockIdx.x + blockIdx.y * gridDim.x;
	int u_ijp0 = (blockIdx.x + 1)%gridDim.x + blockIdx.y * gridDim.x; //down for u
	int u_ijm0 = (blockIdx.x + gridDim.x - 1)%gridDim.x + blockIdx.y * gridDim.x; //up for u
	int u_ij0p = blockIdx.x + ((blockIdx.y + 1)%gridDim.y) * gridDim.x;             //east for u
	int u_ij0m = blockIdx.x + ((blockIdx.y + gridDim.y - 1)%gridDim.y) * gridDim.x; //west for u

	int p_ij00 = blockIdx.x + blockIdx.y * (gridDim.x+1);
	int p_ijp0 = (blockIdx.x + 1)%(gridDim.x+1) + blockIdx.y * (gridDim.x+1); //down for p
	//int p_ijm0 = (blockIdx.x + gridDim.x)%(gridDim.x+1) + blockIdx.y * (gridDim.x+1); //up for p
	//int p_ij0p = blockIdx.x + ((blockIdx.y + 1)%gridDim.y) *(gridDim.x+1);             //east for p
	//int p_ij0m = blockIdx.x + ((blockIdx.y + gridDim.y - 1)%gridDim.y) * (gridDim.x+1); //west for p

	//UPDATE INLET
	if (blockIdx.y==0 && blockIdx.x==0 && EvenOrOdd==RorB)
	{ 	 //Corner point
		Uresid[u_ij00]= (-U[u_ij00]+ U[u_ijp0])+(-3.0*U[u_ij00]+U[u_ij0p])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}

	if (blockIdx.y>0 && blockIdx.y<(dev_N-2) && blockIdx.x==0 && EvenOrOdd==RorB)
	{	  //Middle points
		Uresid[u_ij00]=(-U[u_ij00]+ U[u_ijp0])+(U[u_ij0m]-2.0*U[u_ij00]+U[u_ij0p])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00];  
	}

	if (blockIdx.y==(dev_N-2) && blockIdx.x==0 && EvenOrOdd==RorB)
	{	//Corner point 
		Uresid[u_ij00]= (-U[u_ij00]+ U[u_ijp0])+(U[u_ij0m]-3.0*U[u_ij00])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}

	//UPDATE BULK
	if (blockIdx.y==0 && blockIdx.x>0 && blockIdx.x<(dev_N-1)&& EvenOrOdd==RorB)
	{ // boundary condition
		Uresid[u_ij00]= (U[u_ijm0]-2.0*U[u_ij00]+ U[u_ijp0])+(-3.0*U[u_ij00]+U[u_ij0p])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}
	if (blockIdx.y>0 && blockIdx.y<(dev_N-2) && blockIdx.x>0 && blockIdx.x<(dev_N-1)&& EvenOrOdd==RorB)
	{ //interior
		Uresid[u_ij00]= (U[u_ijm0]-2.0*U[u_ij00]+ U[u_ijp0])+(U[u_ij0m]-2.0*U[u_ij00]+U[u_ij0p])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}
	if (blockIdx.y==(dev_N-2) && blockIdx.x>0 && blockIdx.x<(dev_N-1)&& EvenOrOdd==RorB)
	{ //boundary condition
		Uresid[u_ij00]= (U[u_ijm0]-2.0*U[u_ij00]+ U[u_ijp0])+(U[u_ij0m]-3.0*U[u_ij00])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}

	//Update Outlet
	if (blockIdx.y==0 && blockIdx.x==(dev_N-1)&& EvenOrOdd==RorB)
	{ //boundary condition
		Uresid[u_ij00]= (U[u_ijm0]-U[u_ij00])+(-3.0*U[u_ij00]+U[u_ij0p])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}
	if (blockIdx.y>0 && blockIdx.y<(dev_N-2) && blockIdx.x==(dev_N-1)&& EvenOrOdd==RorB)
	{ //middle points on outlet
		Uresid[u_ij00]= (U[u_ijm0]-U[u_ij00])+(U[u_ij0m]-2.0*U[u_ij00]+U[u_ij0p])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}
	if (blockIdx.y==(dev_N-2) && blockIdx.x==(dev_N-1)&& EvenOrOdd==RorB)
	{ //boundary node
		Uresid[u_ij00]= (U[u_ijm0]-U[u_ij00])+(U[u_ij0m]-3.0*U[u_ij00])-*FAC1*(P[p_ijp0]-P[p_ij00]);
		U[u_ij00]=U[u_ij00]+*OMEGA*Uresid[u_ij00]; 
	}

	__syncthreads();

}


__global__ void update_v(double* V, double* Vresid, double* P, double* Presid, double* FAC1, double* OMEGA, int RorB)
{
	int EvenOrOdd=(blockIdx.x+blockIdx.y)%2;

	int v_ij00 = blockIdx.x + blockIdx.y * (gridDim.x-1);
	int v_ijp0 = (blockIdx.x + 1)%(gridDim.x-1) + blockIdx.y * (gridDim.x-1); //down for v
	int v_ijm0 = (blockIdx.x + gridDim.x - 2)%(gridDim.x-1) + blockIdx.y * (gridDim.x-1); //up for v
	int v_ij0p = blockIdx.x + ((blockIdx.y + 1)%gridDim.y) * (gridDim.x-1);             //east for v
	int v_ij0m = blockIdx.x + ((blockIdx.y + gridDim.y - 1)%gridDim.y) * (gridDim.x-1); //west for v

	//int p_ij00 = blockIdx.x + blockIdx.y * (gridDim.x+1);
	int p_ijp0 = (blockIdx.x + 1)%(gridDim.x+1) + blockIdx.y * (gridDim.x+1); //down for p
	//int p_ijm0 = (blockIdx.x + gridDim.x)%(gridDim.x+1) + blockIdx.y * (gridDim.x+1); //up for p
	//int p_ij0p = blockIdx.x + ((blockIdx.y + 1)%gridDim.y) *(gridDim.x+1);             //east for p
	//int p_ij0m = blockIdx.x + ((blockIdx.y + gridDim.y - 1)%gridDim.y) * (gridDim.x+1); //west for p
	int p_ijpm = (blockIdx.x + 1)%(gridDim.x+1) + ((blockIdx.y + gridDim.y - 1)%gridDim.y) * (gridDim.x+1); //sw for p
	
	//Update inlet similarly to above
	if (blockIdx.y==0 && blockIdx.x==0 && EvenOrOdd==RorB)
	{   //no velocity boundary condition
		Vresid[v_ij00]= 0.0;
		V[v_ij00]=0.0;	
	}

	if (blockIdx.y>0 && blockIdx.y<(dev_N-1) && blockIdx.x==0 && EvenOrOdd==RorB)
	{	  
		Vresid[v_ij00]=(-V[v_ij00]+ V[v_ijp0])+(V[v_ij0m]-2.0*V[v_ij00]+V[v_ij0p])-*FAC1*(P[p_ijp0]-P[p_ijpm]);
		V[v_ij00]=V[v_ij00]+*OMEGA*Vresid[v_ij00];  
	}

	if (blockIdx.y==(dev_N-1) && blockIdx.x==0 && EvenOrOdd==0)
	{	  
		Vresid[v_ij00]= 0.0;
		V[v_ij00]=0.0;	 
	}

	//Update Bulk similarly to above
	if (blockIdx.y==0 && blockIdx.x>0 && blockIdx.x<(dev_N-2)&& EvenOrOdd==RorB)
	{
		Vresid[v_ij00]= 0.0;
		V[v_ij00]=0.0;
	}
	if (blockIdx.y>0 && blockIdx.y<(dev_N-1) && blockIdx.x>0 && blockIdx.x<(dev_N-2)&& EvenOrOdd==RorB)
	{
		Vresid[v_ij00]=(V[v_ijm0]-2.0*V[v_ij00]+ V[v_ijp0])+(V[v_ij0m]-2.0*V[v_ij00]+V[v_ij0p])-*FAC1*(P[p_ijp0]-P[p_ijpm]);
		V[v_ij00]=V[v_ij00]+*OMEGA*Vresid[v_ij00];
	}

	if (blockIdx.y==(dev_N-1) && blockIdx.x>0 && blockIdx.x<(dev_N-2)&& EvenOrOdd==RorB)
	{
		Vresid[v_ij00]=0.0;
		V[v_ij00]=0.0;
	}

	//Update Outlet
	if (blockIdx.y==0 && blockIdx.x==(dev_N-2)&& EvenOrOdd==RorB)
	{
		Vresid[v_ij00]= 0.0;
		V[v_ij00]=0.0;
	}
	if (blockIdx.y>0 && blockIdx.y<(dev_N-1) &&  blockIdx.x==(dev_N-2)&& EvenOrOdd==RorB)
	{
		Vresid[v_ij00]=(V[v_ijm0]-V[v_ij00])+(V[v_ij0m]-2.0*V[v_ij00]+V[v_ij0p])-*FAC1*(P[p_ijp0]-P[p_ijpm]);
		V[v_ij00]=V[v_ij00]+*OMEGA*Vresid[v_ij00];
	}
	if (blockIdx.y==(dev_N-1) && blockIdx.x==(dev_N-2)&& EvenOrOdd==RorB)
	{
		Vresid[v_ij00]= 0.0;
		V[v_ij00]=0.0;
	}
	__syncthreads();

}

__global__ void update_p(double* U, double* V, double* P, double* Presid, double* FAC1, double* OMEGA, double* Pdiff, int RorB)
{
	int EvenOrOdd=((int) (blockIdx.x+blockIdx.y)%2);

	int u_ij00 = blockIdx.x + blockIdx.y * (gridDim.x-1);
	int u_ijm0 = (blockIdx.x + gridDim.x - 2)%(gridDim.x-1) + blockIdx.y * (gridDim.x-1); //up for u

	int v_ijm0 = (blockIdx.x + gridDim.x - 3)%(gridDim.x-2) + blockIdx.y * (gridDim.x-2); //up for v
	int v_ijmp = (blockIdx.x + gridDim.x - 3)%(gridDim.x-2) + ((blockIdx.y + 1)%gridDim.y) * (gridDim.x-2);

	int p_ij00 = blockIdx.x + blockIdx.y * (gridDim.x);
	int p_ijp0 = (blockIdx.x + 1)%(gridDim.x) + blockIdx.y * (gridDim.x); //down for p
	int p_ijm0 = (blockIdx.x + gridDim.x-1)%(gridDim.x) + blockIdx.y * (gridDim.x); //up for p

	//Update the boundary with the right pressure drop
	if (blockIdx.y<(dev_N-1) && blockIdx.x==0 && EvenOrOdd==RorB)
	{	  
		Presid[p_ij00]=2.0*(*Pdiff)-P[p_ijp0]-P[p_ij00];
		P[p_ij00]=2.0*(*Pdiff)-P[p_ijp0];
	}
	//Update interior nodes
	if (blockIdx.y<(dev_N-1) && blockIdx.x>0 && blockIdx.x<(dev_N) && EvenOrOdd==RorB)
	{	  
		Presid[p_ij00]=-(U[u_ij00]-U[u_ijm0])-(V[v_ijmp]-V[v_ijm0]);
		P[p_ij00]=P[p_ij00]+*OMEGA*Presid[p_ij00];
	}
	//Update boundary conditions
	if (blockIdx.y<(dev_N-1) && blockIdx.x==(dev_N) && EvenOrOdd==RorB)
	{	  
		P[p_ij00]=-P[p_ijm0];
	}


	__syncthreads();
}
