#include"global.h"
#include"header.h"

__shared__  int d_dimW[2];


/*__global__ void fdm_acoustic(dx,dt,d_velocity,d_field1,d_field2,bc);
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	
}*/

__device__ int ind(int i, int j, int m, int n)
{
	return i+j*m;
}


__global__ void wavefieldTransfer(float* d_laplace_temp, float* d_field2, int m, int n, int O)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	
	
	if (row >= m || col >= n)return;
	
	
	d_laplace_temp[ind(row+O,col+O,m+2*O,n+2*O)]=d_field2[ind(row,col,m,n)];
	
	
}


__global__ void myCudaMemset(float* A, float val, int m, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	
	
	if (row >= m || col >= n)return;


	A[ind(row,col,m,n)]=val;

	

}




__global__ void calculateLaplace(float* d_laplace,float* d_wave_propagate_t, float* A /*d_laplace_temp*/, float* V,float* d_field1,  float* d_field2, float dt, float dx, int m, int n, float bc, int O, float* C)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row >= m || col >= n)return;

	//O=O+1;

	//if(col>=0 && col <11) printf("C[%d]=%f\n", col, C[col]);
	// printf("dx=%f",dx);
	
	d_laplace[ind(row,col,m,n)]=((C[0] *A[ind(row+O   ,col+O+5 ,m+2*O,n+2*O)])+
								 (C[1] *A[ind(row+O   ,col+O+4 ,m+2*O,n+2*O)])+
								 (C[2] *A[ind(row+O   ,col+O+3 ,m+2*O,n+2*O)])+
								 (C[3] *A[ind(row+O   ,col+O+2 ,m+2*O,n+2*O)])+
								 (C[4] *A[ind(row+O   ,col+O+1 ,m+2*O,n+2*O)])+
								 (C[5] *A[ind(row+O   ,col+O   ,m+2*O,n+2*O)])+
								 (C[6] *A[ind(row+O   ,col+O-1 ,m+2*O,n+2*O)])+
								 (C[7] *A[ind(row+O   ,col+O-2 ,m+2*O,n+2*O)])+
								 (C[8] *A[ind(row+O   ,col+O-3 ,m+2*O,n+2*O)])+
								 (C[9] *A[ind(row+O   ,col+O-4 ,m+2*O,n+2*O)])+
								 (C[10]*A[ind(row+O   ,col+O-5 ,m+2*O,n+2*O)]))/(dx*dx)
								+
								((C[0] *A[ind(row+O+5 ,col+O   ,m+2*O,n+2*O)])+
								 (C[1] *A[ind(row+O+4 ,col+O   ,m+2*O,n+2*O)])+
								 (C[2] *A[ind(row+O+3 ,col+O   ,m+2*O,n+2*O)])+
								 (C[3] *A[ind(row+O+2 ,col+O   ,m+2*O,n+2*O)])+
								 (C[4] *A[ind(row+O+1 ,col+O   ,m+2*O,n+2*O)])+
								 (C[5] *A[ind(row+O   ,col+O   ,m+2*O,n+2*O)])+
								 (C[6] *A[ind(row+O-1 ,col+O   ,m+2*O,n+2*O)])+
								 (C[7] *A[ind(row+O-2 ,col+O   ,m+2*O,n+2*O)])+
								 (C[8] *A[ind(row+O-3 ,col+O   ,m+2*O,n+2*O)])+
								 (C[9] *A[ind(row+O-4 ,col+O   ,m+2*O,n+2*O)])+
								 (C[10]*A[ind(row+O-5 ,col+O   ,m+2*O,n+2*O)]))/(dx*dx);







	/*(A[ind(row+O+1,col+O,m+2*O,n+2*O)]-2*A[ind(row+O,col+O,m+2*O,n+2*O)]+A[ind(row+O-1,col+O,m+2*O,n+2*O)])/(dx*dx)+
								(A[ind(row+O,col+O+1,m+2*O,n+2*O)]-2*A[ind(row+O,col+O,m+2*O,n+2*O)]+A[ind(row+O,col+O-1,m+2*O,n+2*O)])/(dx*dx);




								/*((C[0] *A[ind(row+O   ,col+O   ,m+2*O,n+2*O)])+
								 (C[1] *A[ind(row+O   ,col+O+1 ,m+2*O,n+2*O)])+
								 (C[2] *A[ind(row+O   ,col+O+2 ,m+2*O,n+2*O)])+
								 (C[3] *A[ind(row+O   ,col+O+3 ,m+2*O,n+2*O)])+
								 (C[4] *A[ind(row+O   ,col+O+4 ,m+2*O,n+2*O)])+
								 (C[5] *A[ind(row+O   ,col+O+5 ,m+2*O,n+2*O)])+
								 (C[6] *A[ind(row+O   ,col+O+6 ,m+2*O,n+2*O)])+
								 (C[7] *A[ind(row+O   ,col+O+7 ,m+2*O,n+2*O)])+
								 (C[8] *A[ind(row+O   ,col+O+8 ,m+2*O,n+2*O)])+
								 (C[9] *A[ind(row+O   ,col+O+9 ,m+2*O,n+2*O)])+
								 (C[10]*A[ind(row+O   ,col+O+10,m+2*O,n+2*O)]))/(dx*dx)
								+
								((C[0] *A[ind(row+O   ,col+O   ,m+2*O,n+2*O)])+
								 (C[1] *A[ind(row+O+1 ,col+O   ,m+2*O,n+2*O)])+
								 (C[2] *A[ind(row+O+2 ,col+O   ,m+2*O,n+2*O)])+
								 (C[3] *A[ind(row+O+3 ,col+O   ,m+2*O,n+2*O)])+
								 (C[4] *A[ind(row+O+4 ,col+O   ,m+2*O,n+2*O)])+
								 (C[5] *A[ind(row+O+5 ,col+O   ,m+2*O,n+2*O)])+
								 (C[6] *A[ind(row+O+6 ,col+O   ,m+2*O,n+2*O)])+
								 (C[7] *A[ind(row+O+7 ,col+O   ,m+2*O,n+2*O)])+
								 (C[8] *A[ind(row+O+8 ,col+O   ,m+2*O,n+2*O)])+
								 (C[9] *A[ind(row+O+9 ,col+O   ,m+2*O,n+2*O)])+
								 (C[10]*A[ind(row+O+10,col+O   ,m+2*O,n+2*O)]))/(dx*dx);




	/*(5.859325396829981f*A[ind(row+O-1,col+O-1,m+O,n+O)]-27.485714285746056*A[ind(row+O,col+O-1,m+O,n+O)]+
		62.100000000097751*A[ind(row+O+1,col+O-1,m+O,n+O)]-89.022222222396508*A[ind(row+O+2,col+O-1,m+O,n+O)]+
		86.375000000196621*A[ind(row+O+3,col+O-1,m+O,n+O)]-56.400000000143436*A[ind(row+O+4,col+O-1,m+O,n+O)]+
		23.811111111177066*A[ind(row+O+5,col+O-1,m+O,n+O)]-5.885714285731734*A[ind(row+O+6,col+O-1,m+O,n+O)]+
		0.648214285716317*A[ind(row+O+7,col+O-1,m+O,n+O)])/(dx*dx);/*+
	(5.859325396829981f*A[ind(row+O-1,col+O-1,m+O,n+O)]-27.485714285746056*A[ind(row+O-1,col+O,m+O,n+O)]+
		62.100000000097751*A[ind(row+O-1,col+O+1,m+O,n+O)]-89.022222222396508*A[ind(row+O-1,col+O+2,m+O,n+O)]+
		86.375000000196621*A[ind(row+O-1,col+O+3,m+O,n+O)]-56.400000000143436*A[ind(row+O-1,col+O+4,m+O,n+O)]+
		23.811111111177066*A[ind(row+O-1,col+O+5,m+O,n+O)]-5.885714285731734*A[ind(row+O-1,col+O+6,m+O,n+O)]+
		0.648214285716317*A[ind(row+O-1,col+O+7,m+O,n+O)])/(dx*dx);*/



//	(A[ind(row+2,col+1,m+2,n+2)]-2*A[ind(row+1,col+1,m+2,n+2)]+A[ind(row,col+1,m+2,n+2)])/(dx*dx)+
//								(A[ind(row+1,col+2,m+2,n+2)]-2*A[ind(row+1,col+1,m+2,n+2)]+A[ind(row+1,col,m+2,n+2)])/(dx*dx);
	
	
	d_wave_propagate_t[ind(row,col,m,n)]=(V[ind(row,col,m,n)]*V[ind(row,col,m,n)])*(dt*dt)*d_laplace[ind(row,col,m,n)]+
										2*d_field2[ind(row,col,m,n)]-d_field1[ind(row,col,m,n)];
										
	
	
	if(bc==1)
	{
		d_wave_propagate_t[ind(row,0,m,n)]=0;
		d_wave_propagate_t[ind(row,n-1,m,n)]=0;
		d_wave_propagate_t[ind(1,col,m,n)]=0;
		d_wave_propagate_t[ind(m-1,col,m,n)]=0;		
	}
	else
	{
		d_wave_propagate_t[ind(row,0,m,n)]=0;
		d_wave_propagate_t[ind(row,n-1,m,n)]=0;
		d_wave_propagate_t[ind(m-1,col,m,n)]=0;		
	}
}


__global__ void add_source(float* d_wave_propagate_t, float* d_field2, float* source_grid,float w, int m, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	
	if (row >= m || col >= n)return;
	
	d_field2[ind(row,col,m,n)]=d_wave_propagate_t[ind(row,col,m,n)]+source_grid[ind(row,col,m,n)]*w;
}

__global__ void ABC_inner(float* V,float* f1, float* f2, float* wave, float dx, float dt, float bc,int m, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row >= m || col >= n)return;
	
	if(bc==1)
	{
		//Applying Bounday condition for Top
		if(row==1 && col<n-6)
		{
			 wave[ind(1,col+3,m,n)] = (2*V[ind(1,col+3,m,n)]*dx*dt*dt)/(dx+V[ind(1,col+3,m,n)]*dt)*
         ((wave[ind(2,col+3,m,n)]/(2*dt*dx) - f1[ind(2,col+3,m,n)]/(2*dx*dt)+
         f1[ind(1,col+3,m,n)]/(2*dt*dx))+
    (-1/(2*dt*dt*V[ind(1,col+3,m,n)]))*
         (-2*f2[ind(1,col+3,m,n)] + f1[ind(1,col+3,m,n)] -2*f2[ind(2,col+3,m,n)]+
         f1[ind(2,col+3,m,n)] + wave[ind(2,col+3,m,n)]) +
    V[ind(1,col+3,m,n)]/(4*dx*dx)*
         (wave[ind(2,col+4,m,n)] + f1[ind(1,col+4,m,n)] + wave[ind(2,col+2,m,n)] -
          2* wave[ind(2,col+3,m,n)] - 2*f1[ind(1,col+3,m,n)] + f1[ind(1,col+2,m,n)]));
		}
	}
	
	//Applying Bounday condition for Bottom
	
	if(row==1 && col<n-6)
	{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
		wave[ind(m-2,col+3,m,n)] = -2.*dx*dt*dt*V[ind(m-2,col+3,m,n)]/(dx+V[ind(m-2,col+3,m,n)]*dt)*
       ((-wave[ind(m-3,col+3,m,n)]/(2*dt*dx) - f1[ind(m-2,col+3,m,n)]/(2*dt*dx) +
       f1[ind(m-3,col+3,m,n)]/(2*dt*dx)) +
  1/(2*dt*dt*V[ind(m-2,col+3,m,n)])*
       (-2*f2[ind(m-2,col+3,m,n)] + f1[ind(m-2,col+3,m,n)] + wave[ind(m-3,col+3,m,n)] - 
       2*f2[ind(m-3,col+3,m,n)] + f1[ind(m-3,col+3,m,n)]) + 
  (-V[ind(m-2,col+3,m,n)]/(4*dx*dx))* 
       (wave[ind(m-3,col+4,m,n)] - 2*wave[ind(m-3,col+3,m,n)] + wave[ind(m-3,col+2,m,n)] + 
       f1[ind(m-2,col+4,m,n)] - 2*f1[ind(m-2,col+3,m,n)] + f1[ind(m-2,col+2,m,n)]));
	}

	//Applying Boundary condition for right hand side
	if(row<m-6 && col==n-1)
	{
		wave[ind(row+3,n-2,m,n)] =  -2*dx*dt*dt*V[ind(row+3,n-2,m,n)]/(dx+V[ind(row+3,n-2,m,n)]*dt)*
       ((-wave[ind(row+3,n-3,m,n)]/(2*dt*dx) - f1[ind(row+3,n-2,m,n)]/(2*dt*dx) + 
       f1[ind(row+3,n-3,m,n)]/(2*dt*dx)) + 
  1/(2*dt*dt*V[ind(row+3,n-2,m,n)])*
       (-2*f2[ind(row+3,n-2,m,n)] + f1[ind(row+3,n-2,m,n)] + wave[ind(row+3,n-3,m,n)] - 
       2*f2[ind(row+3,n-3,m,n)] + f1[ind(row+3,n-3,m,n)]) + 
  (-V[ind(row+3,n-2,m,n)]/(4*dx*dx))* 
       (wave[ind(row+4,n-3,m,n)] - 2*wave[ind(row+3,n-3,m,n)] + wave[ind(row+2,n-3,m,n)] + 
       f1[ind(row+4,n-2,m,n)] - 2*f1[ind(row+3,n-2,m,n)] + f1[ind(row+2,n-2,m,n)]));
	}

	//Applying Boundary condition for left hand side
	if(row<m-6 && col==1)
	{
		wave[ind(row+3,1,m,n)] =(2*V[ind(row+3,1,m,n)]*dx*dt*dt)/(dx+V[ind(row+3,1,m,n)]*dt)*
       ((wave[ind(row+3,2,m,n)]/(2*dt*dx) - f1[ind(row+3,2,m,n)]/(2*dt*dx) +
       f1[ind(row+3,1,m,n)]/(2*dt*dx)) + 
  (-1/(2*dt*dt*V[ind(row+3,1,m,n)]))*
       (-2*f2[ind(row+3,1,m,n)] + f1[ind(row+3,1,m,n)] + wave[ind(row+3,2,m,n)] -
       2*f2[ind(row+3,2,m,n)] + f1[ind(row+3,2,m,n)]) +
  (V[ind(row+3,1,m,n)]/(4*dx*dx))*
       (wave[ind(row+4,2,m,n)] - 2*wave[ind(row+3,2,m,n)] + wave[ind(row+2,2,m,n)] +
       f1[ind(row+4,1,m,n)] - 2*f1[ind(row+3,1,m,n)] + f1[ind(row+2,1,m,n)]));
	}
	
	//Applying Lower Right hand corner
	if(row==1 && col==1)
	{
		
		wave[ind(m-3,n-2,m,n)] = V[ind(m-3,n-2,m,n)]*dt*dx/(2*V[ind(m-3,n-2,m,n)]*dt + sqrt(2.)*dx)*
       (wave[ind(m-4,n-2,m,n)]/dx + wave[ind(m-3,n-3,m,n)]/dx + 
       sqrt(2.)/(V[ind(m-3,n-2,m,n)]*dt)*f2[ind(m-3,n-2,m,n)]);

wave[ind(m-2,n-3,m,n)] = V[ind(m-2,n-3,m,n)]*dt*dx/(2*V[ind(m-2,n-3,m,n)]*dt + sqrt(2.)*dx)*
       (wave[ind(m-3,n-3,m,n)]/dx + wave[ind(m-2,n-4,m,n)]/dx + 
       sqrt(2.)/(V[ind(m-2,n-3,m,n)]*dt)*f2[ind(m-2,n-3,m,n)]);

wave[ind(m-2,n-2,m,n)] = V[ind(m-2,n-2,m,n)]*dt*dx/(2*V[ind(m-2,n-2,m,n)]*dt + sqrt(2.)*dx)*
       (wave[ind(m-3,n-2,m,n)]/dx + wave[ind(m-2,n-3,m,n)]/dx + 
       sqrt(2.)/(V[ind(m-2,n-2,m,n)]*dt)*f2[ind(m-2,n-2,m,n)]);
		
	}

	//Applying Lower Left hand corner
	if(row==5 && col==5)
	{
		wave[ind(m-3,1,m,n)] = V[ind(m-3,1,m,n)]*dt*dx/(2*V[ind(m-3,1,m,n)]*dt + sqrt(2.)*dx)*
       (wave[ind(m-4,1,m,n)]/dx + wave[ind(m-3,2,m,n)]/dx +
       sqrt(2.)/(V[ind(m-3,1,m,n)]*dt)*f2[ind(m-3,1,m,n)]);

wave[ind(m-2,2,m,n)] = V[ind(m-2,2,m,n)]*dt*dx/(2*V[ind(m-2,2,m,n)]*dt + sqrt(2.)*dx)*
       (wave[ind(m-3,2,m,n)]/dx + wave[ind(m-2,3,m,n)]/dx +
       sqrt(2.)/(V[ind(m-2,2,m,n)]*dt)*f2[ind(m-2,2,m,n)]);

wave[ind(m-2,1,m,n)] = V[ind(m-2,1,m,n)]*dt*dx/(2*V[ind(m-2,1,m,n)]*dt + sqrt(2.)*dx)*
       (wave[ind(m-3,1,m,n)]/dx + wave[ind(m-2,2,m,n)]/dx +
       sqrt(2.)/(V[ind(m-2,1,m,n)]*dt)*f2[ind(m-2,1,m,n)]);
	}
	

if(bc==1)
{

  // for upper right hand corner

  wave[ind(2,n-2,m,n)] = V[ind(2,n-2,m,n)]*dt*dx/(2*V[ind(2,n-2,m,n)]*dt + sqrt(2.)*dx)*
         (wave[ind(3,n-2,m,n)]/dx + wave[ind(2,n-3,m,n)]/dx +
         sqrt(2.)/(V[ind(2,n-2,m,n)]*dt)*f2[ind(2,n-2,m,n)]); 

  wave[ind(1,n-3,m,n)] = V[ind(1,n-3,m,n)]*dt*dx/(2*V[ind(1,n-3,m,n)]*dt + sqrt(2.)*dx)*
         (wave[ind(2,n-3,m,n)]/dx + wave[ind(1,n-4,m,n)]/dx +
         sqrt(2.)/(V[ind(1,n-3,m,n)]*dt)*f2[ind(1,n-3,m,n)]);

  wave[ind(1,n-2,m,n)] = V[ind(1,n-2,m,n)]*dt*dx/(2*V[ind(1,n-2,m,n)]*dt + sqrt(2.)*dx)*
         (wave[ind(2,n-2,m,n)]/dx + wave[ind(1,n-3,m,n)]/dx +
         sqrt(2.)/(V[ind(1,n-2,m,n)]*dt)*f2[ind(1,n-2,m,n)]);

  // for upper left hand corner
  
  wave[ind(2,1,m,n)] = V[ind(2,1,m,n)]*dt*dx/(2*V[ind(2,1,m,n)]*dt + sqrt(2.)*dx)*
         (wave[ind(3,1,m,n)]/dx + wave[ind(2,2,m,n)]/dx +
         sqrt(2.)/(V[ind(2,1,m,n)]*dt)*f2[ind(2,1,m,n)]);
  
  wave[ind(1,2,m,n)] = V[ind(1,2,m,n)]*dt*dx/(2*V[ind(1,2,m,n)]*dt + sqrt(2.)*dx)*
         (wave[ind(2,2,m,n)]/dx + wave[ind(1,3,m,n)]/dx +
         sqrt(2.)/(V[ind(1,2,m,n)]*dt)*f2[ind(1,2,m,n)]); 

  wave[ind(1,1,m,n)] = V[ind(1,1,m,n)]*dt*dx/(2*V[ind(1,1,m,n)]*dt + sqrt(2.)*dx)*
         (wave[ind(2,1,m,n)]/dx + wave[ind(1,2,m,n)]/dx +
         sqrt(2.)/(V[ind(1,1,m,n)]*dt)*f2[ind(1,1,m,n)]);

}
	
}

__global__ void ABC_outer(float* d_velocity,float* d_field1, float* d_field2, float* d_wave_propagate_t, float dx, float dt, float bc,int m, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row >= m || col >= n)return;
	
	
	
}

__global__ void rickerWavelet(float* w, float f, float n, float dt)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int id=row*n+col;

	//float T=dt*(n-1);
	float t0=1/f;
	
	if(id>=n) return;

	float t=id*dt;
	float tau=t-t0;


	w[id]=(1-(tau*tau*f*f*PI*PI))*expf(-tau*tau*PI*f*PI*f);
	//if (row==1) printf("PI=%f\n",PI);

	

}

__global__ void extractCorrectRegion(float* A,float* B, int m, int n, int p)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= m || col >= n)return;

	A[ind(row,col,m,n)]=B[ind(row+p,col+p,m+2*p,n+2*p)];
}

__global__ void badBoundaryCondition(float* A, float* T, int m, int n, int p)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= m || col >= n)return;

	// First Zeroing matrix A
	T[ind(row+p,col+p,m+2*p,n+2*p)]=A[ind(row+p,col+p,m+2*p,n+2*p)];


}

__global__ void calculateCerjanCoeff(int p, float* G, int I)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int id=row*blockDim.y*gridDim.y+col;

	if(id>=p) return;

	float a=0.0049*(I-id);
	G[id]=expf(-a*a);

	// if(id<I-1) G[id]=1.0;
}

__global__ void cerjanMatrix(float* A, float* G, int m, int n, int p)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row >= m || col >= n)return;

	A[ind(row,col,m,n)]=1;

	if(row< p)	A[ind(row,col,m,n)]=A[ind(row,col,m,n)]*G[p-row-1];

	if(row>=m-p) A[ind(row,col,m,n)]=A[ind(row,col,m,n)]*G[row-(m-p)];

	if(col<p) A[ind(row,col,m,n)]=A[ind(row,col,m,n)]*G[p-col-1];

	if(col>=n-p) A[ind(row,col,m,n)]=A[ind(row,col,m,n)]*G[col-(n-p)];
	


}

__global__ void cerjanBoundaryCondition(float* A, float* B, float* C, float* CM, int m, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row >= m || col >= n)return;

	A[ind(row,col,m,n)]=A[ind(row,col,m,n)]*CM[ind(row,col,m,n)];
	B[ind(row,col,m,n)]=B[ind(row,col,m,n)]*CM[ind(row,col,m,n)];
	C[ind(row,col,m,n)]=C[ind(row,col,m,n)]*CM[ind(row,col,m,n)];


}