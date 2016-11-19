#include"global.h"
#include"header.h"

using namespace std;


float max(float a, float b)
{
	if(a>b)
		return a;
	else
		return b;
}


float max( float* A , int dimA[])
{
	float max1=A[0];
	for(int i=1; i<dimA[0]*dimA[1]; i++)
	{
		max1=max(A[i],max1);
	}
	
	return max1;
}

float max( float* A , int dimA)
{
	float max1=A[0];
	for(int i=1; i<dimA; i++)
	{
		max1=max(A[i],max1);
	}
	
	return max1;
}

float* consVec(float *A, int dimA, float a)
{
	for(int i=0;i<dimA;i++)
	{
		A[i]=a;
	}
	return A;
}

int ind(int i, int j)
{
	return i+j*dimMod[0];
}

void copy1Dto1D(float* A, float* B, int dimA)
{
	for(int i=0; i<dimA; i++)
		B[i]=A[i];
}




