#include"global.h"
#include"header.h"

 


void readASCIIFile(char *fn, int dimA[2], float* A)
{


	FILE *fid;
	fid=fopen(fn,"r");
	
	if(fid==NULL) perror ("Error opening file");
	
	
	for(int i=0; i<dimA[0]*dimA[1]; i++)
	{
		fscanf(fid,"%f",A+i);
	}

}

int VectoFileWrite(float* A, int M, const char* fn)
{
	FILE * pFile;
   	
   	pFile = fopen (fn,"w+");
   	for (int i=0 ; i<M ; i++)
   	{
		fprintf (pFile, "%f\n",A[i]);		
	
   	}
   	fclose (pFile);
	printf("%s Written!\n",fn);
	return 1;
   	
}