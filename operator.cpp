#include"global.h"
#include"header.h"

using namespace std;

int ind1(int i, int j, int m, int n)
{
	return i+j*m;
}


void construct_source(float* field1, float* field2, float w1)
{
	float x_max=(NX/*dimMod[1]*/-1)*dx,
		z_max=(NZ/*dimMod[0]*/-1)*dz,
		bin_x=floor(source_x/dx),
		bin_z=floor(source_z/dz);
		
	int m=(int)bin_z,
		n=(int)bin_x;
		
	//printf("m=%d,\nn=%d\n",m,n);
		
	if(remainderf(source_x,dx)==0)
	{
		if(remainderf(source_z,dz)==0)
		{
			if(source_x!=x_max && source_z!=z_max)
			{
				// field1[ind(m,n+1)]=1*w1;
				// field1[ind(m,n-1)]=1*w1;
				// field1[ind(m-1,n)]=1*w1;
				// field1[ind(m+1,n)]=1*w1;
				field1[ind(m,n)]=1*w1;
				
				
			}
			
		}
		else
		{
			if(source_x!=x_max)
			{
				field1[ind(m,n)]=1*w1;
				// field1[ind(m,n+1)]=1*w1;
				// field1[ind(m+1,n)]=1*w1;
				// field1[ind(m+1,n+1)]=1*w1;
				
			}
		}		
	}
	else
	{
		field1[ind(m,n)]=1*w1;
		// field1[ind(m+1,n)]=1*w1;
		// field1[ind(m,n+1)]=1*w1;
		// field1[ind(m+1,n+1)]=1*w1;
	}
	
	if(remainderf(source_x,dx)!=0)
	{
		if(remainderf(source_x,dx)==0)
		{
			if(source_z!=z_max)
			{
				field1[ind(m,n)]=1*w1;
				// field1[ind(m+1,n)]=1*w1;
				// field1[ind(m,n+1)]=1*w1;
				// field1[ind(m+1,n+1)]=1*w1;
			}
		}
	}
	
	if(source_x==x_max)
	{
		field1[ind(m,n)]=1*w1;
		// field1[ind(m+1,n)]=1*w1;
		// field1[ind(m+1,n-1)]=1*w1;
		// field1[ind(m,n-1)]=1*w1;
	}
	
	if(source_z==z_max)
	{
		field1[ind(m,n)]=1*w1;
		// field1[ind(m-1,n)]=1*w1;
		// field1[ind(m,n+1)]=1*w1;
		// field1[ind(m-1,n+1)]=1*w1;
	}
		
		
	copy1Dto1D(field1, field2, NX*NZ/*dimMod[0]*dimMod[1]*/);
	
}


void cudaCheck(cudaError_t cudaStatus)
{
	if(cudaStatus!=cudaSuccess)
	{
		printf("Cuda couldn't allocated! Error no.:%d\n",(int)cudaStatus);
		exit(1);
	}
}

void fdm_acoustic(float* d_velocity,float* d_field1,float* d_field2,float bc, dim3 gridDim, dim3 blockDim)
{
	
}

void matrixPadding(float* A, float* B, int m, int n, int p)
{	
	int i,j;
	for(i=0;i<(m+2*p)*(n+2*p);i++) A[i]=0; 
	
	for(i=0; i< m; i++)
		for(j=0; j< n;j++)
			A[ind1(i+p,j+p,m+2*p,n+2*p)]=B[ind1(i,j,m,n)];

	for(i=0;i<p;i++)
	{
		for(j=0;j<m;j++)
		{
			A[ind1(j+p,i,m+2*p,n+2*p)]=A[ind1(j+p,p,m+2*p,n+2*p)];
			A[ind1(j+p,i+p+n,m+2*p,n+2*p)]=A[ind1(j+p,p+n-1,m+2*p,n+2*p)];
		}

		for(j=0;j<n;j++)
		{
			A[ind1(i,j+p,m+2*p,n+2*p)]=A[ind1(p,j+p,m+2*p,n+2*p)];
			A[ind1(i+m+p,j+p,m+2*p,n+2*p)]=A[ind1(m+p-1,j+p,m+2*p,n+2*p)];
		}

		for(j=0;j<p;j++)
		{
			A[ind1(i,j,m+2*p,n+2*p)]=A[ind1(i,p,m+2*p,n+2*p)];
			A[ind1(i+p+m,j,m+2*p,n+2*p)]=A[ind1(i+p+m,p,m+2*p,n+2*p)];
			A[ind1(i,j+p+n,m+2*p,n+2*p)]=A[ind1(i,p+n-1,m+2*p,n+2*p)];
			A[ind1(i+p+m,j+p+n,m+2*p,n+2*p)]=A[ind1(i+p+m,p+n-1,m+2*p,n+2*p)];
		}
	}
}