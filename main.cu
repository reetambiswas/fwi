#include"global.h"
#include"header.h"


using namespace std;



int main()
{
	pad=60;
	O=10;

	//Define Dimensions of model
	dimMod[0]=122;dimMod[1]=384;

	NX=dimMod[1]+2*pad;
	NZ=dimMod[0]+2*pad;
	NZ1=dimMod[0]+2*pad;


	

	//Parameters for the modelling
	float t_n=3;
	float f=15;
	dt=0.001;
	dx=10;
	dz=dx;
	

	int nt=(int)t_n/dt;

	//Define File Names
	// std::string veloFn="marmousi.dat";
	// std::string veloFn="marmsmooth.dat";
	// std::string veloFn="marmhard.dat";
	std::string veloFn="homogenous.velo";
	// std::string sourceFn="wavelet2.dat";
	
	

	float velo1[dimMod[0]*dimMod[1]],
	field1[NX*NZ],//[dimMod[0]*dimMod[1]],
	field2[NX*NZ],//[dimMod[0]*dimMod[1]],
	velocity[NZ*NX];
	
	consVec(field1, NX*NZ,/*dimMod[0]*dimMod[1]*/ 0);
	consVec(field2, NX*NZ,/*dimMod[0]*dimMod[1]*/ 0);

	
	ofstream myFile ("wavefield1.bin", ios::out | ios::binary);
	
	
	dimW[0]=nt;dimW[1]=1;
	float wavelet[dimW[0]*dimW[1]];
	
	
	
	//Read files for velocity model, wavelet etc.
	readASCIIFile((char*)veloFn.c_str(),dimMod, velo1);
	matrixPadding(velocity,velo1, dimMod[0], dimMod[1],pad);

	// VectoFileWrite(velocity, NX*NZ, "outVelocity.dat");
	// exit(1);
	dx=max(velo1,dimMod)/f/6;
	//readASCIIFile((char*)sourceFn.c_str(),dimW, wavelet);
	
	// source_x=(dimMod[1]/2)*dx;//dimMod[1]/2*dx;
	// source_z=(dimMod[0]/2)*dx;//(30)*dx,
	source_x=(dimMod[1]/2+pad)*dx;
	source_z=(10+pad)*dx,
	bc=1;
	

	// float C[11]={0.565794,-6.261905,31.544643,-95.523810,193.361111,-275.080000,281.291667,-207.650794,109.303571,-38.579365,7.029087};
	//float C[11]={7.029087,-38.579365,109.303571,-207.650794,281.291667,-275.080000,193.361111,-95.523810,31.544643,-6.261905,0.565794};
	float C[11]={0.000317,-0.004960,0.039683,-0.238095,1.666667,-2.927222,1.666667,-0.238095,0.039683,-0.004960,0.000317};
	// float CC[11];
	
	clock_t t1,t2,t3,t4;
	
	

	printf("Position of Source:\nX:%f\nZ:%f\n",source_x,source_z);
	

	double stability=(double)(dt/dx)*max(velo1,dimMod);
	printf("Stability=%f\n",(float)stability);
	
	if(stability>1/sqrt(2))
	{
		printf("Grid not stable! Check the parameters and run again!\n");
		exit(1);		
	}	
	
	
	
	
	//VectoFileWrite(field2, dimMod[0]*dimMod[1], "field2.dat");
	t1=clock();
	//Initializing Cuda Device
	int numDevs= 0;
	cudaGetDeviceCount(&numDevs);
	//For now using only one device
	printf("Number of Device:%d\n",numDevs);
	int deviceID=0;
	cudaSetDevice(deviceID );
	int cDeviceID=0;
	cudaGetDevice(&cDeviceID );
	printf("Current Active Device ID:%d\n",cDeviceID);
	
	dim3 blockDim(32,32);
	//dim3 gridDim((int)ceil(dimMod[1]/32)+1,(int)ceil(dimMod[0]/32)+1);
	dim3 gridDim(20,20);


	printf("GridDim:(%d,%d)\nBlockDim(%d,%d)\n",gridDim.x,gridDim.y,blockDim.x,blockDim.y);
	
	t2=clock();
	//Creating Device variables in cuda devices
	float *d_field1, *d_field2, *d_wavelet, *source_grid, *d_velocity, *d_wave_propagate_t, 
	*d_laplace_temp, *d_laplace, *d_C, *d_correctField, *d_temp1, *d_G, *d_cerjanMatrix;
	cudaError_t cudaStatus;
	
	cudaStatus=cudaMalloc((void**)&d_field1,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float));
	cudaCheck(cudaStatus);
	
	
	cudaStatus=cudaMalloc((void**)&d_field2,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float));
	cudaCheck(cudaStatus);

	cudaStatus=cudaMalloc((void**)&d_temp1,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float));
	cudaCheck(cudaStatus);
	
	cudaStatus=cudaMalloc((void**)&d_wavelet,dimW[0]*dimW[1]*sizeof(float));
	cudaCheck(cudaStatus);
	
	cudaStatus=cudaMalloc((void**)&source_grid,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float));
	cudaCheck(cudaStatus);

	cudaStatus=cudaMalloc((void**)&d_correctField,dimMod[0]*dimMod[1]*sizeof(float));
	cudaCheck(cudaStatus);
	
	// cudaStatus=cudaMalloc((void**)&d_velocity,dimMod[0]*dimMod[1]*sizeof(float));
	// cudaCheck(cudaStatus);

	cudaStatus=cudaMalloc((void**)&d_velocity,NX*NZ*sizeof(float));
	cudaCheck(cudaStatus);
	
	cudaStatus=cudaMalloc((void**)&d_wave_propagate_t,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float));
	cudaCheck(cudaStatus);
	
	cudaStatus=cudaMalloc((void**)&d_laplace_temp,(NZ/*dimMod[0]*/+2*O)*(NX/*dimMod[1]*/+2*O)*sizeof(float));
	cudaCheck(cudaStatus);
	
	cudaStatus=cudaMalloc((void**)&d_laplace,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float));
	cudaCheck(cudaStatus);

	cudaStatus=cudaMalloc((void**)&d_cerjanMatrix,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float));
	cudaCheck(cudaStatus);

	cudaStatus=cudaMalloc((void**)&d_C,(O+1)*sizeof(float));
	cudaCheck(cudaStatus);

	cudaStatus=cudaMalloc((void**)&d_G,(O+pad)*sizeof(float));
	cudaCheck(cudaStatus);
	
	//Generating Wavelet

	rickerWavelet<<<gridDim,blockDim>>>( d_wavelet, f, dimW[0]*dimW[1], dt);
	cudaStatus = cudaMemcpy(wavelet, d_wavelet , dimW[0]*dimW[1]*sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheck(cudaStatus);
	VectoFileWrite(wavelet, dimW[0]*dimW[1], "outWavelet.dat");
	

	//Constructing the initial fields by placing the source in correct position		
	construct_source(field1,field2,wavelet[0]);


	//Copying Data to Device
	cudaStatus = cudaMemcpy(d_field1, field1 , NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheck(cudaStatus);
	
	cudaStatus = cudaMemcpy(d_field2, field2 , NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheck(cudaStatus);

	cudaStatus = cudaMemcpy(d_wavelet, wavelet , dimW[0]*dimW[1]*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheck(cudaStatus);
	
	cudaStatus = cudaMemcpy(source_grid, field2 , NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheck(cudaStatus);
	
	cudaStatus = cudaMemcpy(d_velocity, velocity , NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheck(cudaStatus);

	cudaStatus = cudaMemcpy(d_C, C , (O+1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheck(cudaStatus);

	/*cudaStatus = cudaMemcpy(CC, d_C , O*sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheck(cudaStatus);
	VectoFileWrite(CC, O, "CC.dat");
	*/

	// Calculating Cerjan Boundary Condition Coefficients
	calculateCerjanCoeff<<<gridDim,blockDim>>>(pad, d_G, 60);

	cerjanMatrix<<<gridDim,blockDim>>>(d_cerjanMatrix, d_G, NZ,NX, pad);

	float CM[NX*NZ];
	cudaStatus = cudaMemcpy(CM, d_cerjanMatrix , (NZ*NX)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheck(cudaStatus);
	VectoFileWrite(CM, NX*NZ, "outCerjanMatrix.dat");

	float G[pad+O];
	cudaStatus = cudaMemcpy(G, d_G , (pad)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheck(cudaStatus);

	VectoFileWrite(G, pad, "outCerjan.dat");
	// exit(1);

	int size=dimMod[0]*dimMod[1]; //(dimMod[0])*(dimMod[1]);
	float check[size];
	// int size1=(dimMod[0]+2*O)*(dimMod[1]+2*O);
	// float check1[size1];
	// cudaStatus = cudaMemcpy(check, d_velocity , size*sizeof(float), cudaMemcpyDeviceToHost);
	// if(cudaStatus!=cudaSuccess)
	// {
		// printf("Cuda couldn't allocated! Error no.:%d\n",(int)cudaStatus);
	// }
	// VectoFileWrite(check, size, "check.dat");
	t3=clock();
	printf("Running Loop!\n");
	//char buffer[32];
	for(int step=1; step<=nt; step++)
	{
		//Wave Propagation
		// fdm_acoustic(d_velocity,d_field1,d_field2,bc,gridDim,blockDim);
		
		//Calculate Laplacian
		// myCudaMemset  (  d_laplace_temp,0.0, (dimMod[0]+2*O),(dimMod[1]+2*O)  );      
		myCudaMemset<<<gridDim,blockDim>>>(d_laplace_temp, 0.0, (NZ/*dimMod[0]*/+2*O), (NX/*dimMod[1]*/+2*O));

		//myCudaMemset<<<gridDim,blockDim>>>(  d_field2,1.0, (dimMod[0]),(dimMod[1]));      
		

		// cudaStatus = cudaMemcpy(check, d_field2 , size*sizeof(float), cudaMemcpyDeviceToHost);
		// cudaCheck(cudaStatus);
		// VectoFileWrite(check, size, "field2.dat");
		
		
		//wavefield Transfer
		wavefieldTransfer<<<gridDim,blockDim>>>(d_laplace_temp,d_field2,NZ/*dimMod[0]*/,NX/*dimMod[1]*/,O);
		// cudaStatus = cudaMemcpy(check1, d_laplace_temp , size1*sizeof(float), cudaMemcpyDeviceToHost);
		// cudaCheck(cudaStatus);
		// VectoFileWrite(check1, size1, "check1.dat");
		// exit(1);
		
		//Propagating waves
		calculateLaplace<<<gridDim,blockDim>>>(d_laplace, d_wave_propagate_t, d_laplace_temp, d_velocity,
												d_field1,d_field2,dt,dx, NZ/*dimMod[0]*/,NX/*dimMod[1]*/, bc, O, d_C);
		// cudaStatus = cudaMemcpy(check, d_laplace , size*sizeof(float), cudaMemcpyDeviceToHost);
		// cudaCheck(cudaStatus);
		// VectoFileWrite(check, size, "check.dat");
		//exit(1);
			
		//ABC
		 ABC_inner<<<gridDim,blockDim>>>(d_velocity,d_field1,d_field2,d_wave_propagate_t,dx,dt,bc,NZ/*dimMod[0]*/,NX/*dimMod[1]*/);
		//ABC_outer<<<gridDim,blockDim>>>(d_velocity,d_field1,d_field2,d_wave_propagate_t,dx,dt,bc,dimMod[0],dimMod[1]);
		// BadBoundaryCondition makes everything in the extra region zero
		// myCudaMemset<<<gridDim,blockDim>>>(d_temp1, 0.0, NZ/*dimMod[0]*/, NX);
		// badBoundaryCondition<<<gridDim,blockDim>>>(d_wave_propagate_t,d_temp1,dimMod[0],dimMod[1],pad);
		// cudaStatus=cudaMemcpy(d_wave_propagate_t,d_temp1,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float),cudaMemcpyDeviceToDevice);
		// cudaCheck(cudaStatus);

		// myCudaMemset<<<gridDim,blockDim>>>(d_temp1, 0.0, NZ/*dimMod[0]*/, NX);
		// badBoundaryCondition<<<gridDim,blockDim>>>(d_field1,d_temp1,dimMod[0],dimMod[1],pad);
		// cudaStatus=cudaMemcpy(d_field1,d_temp1,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float),cudaMemcpyDeviceToDevice);
		// cudaCheck(cudaStatus);

		// myCudaMemset<<<gridDim,blockDim>>>(d_temp1, 0.0, NZ/*dimMod[0]*/, NX);
		// badBoundaryCondition<<<gridDim,blockDim>>>(d_field2,d_temp1,dimMod[0],dimMod[1],pad);
		// cudaStatus=cudaMemcpy(d_field2,d_temp1,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float),cudaMemcpyDeviceToDevice);
		// cudaCheck(cudaStatus);

		// cerjanBoundaryCondition<<<gridDim,blockDim>>>( d_wave_propagate_t,  d_field1,  d_field2, d_cerjanMatrix, NZ, NX);

			
			
			
		//field1=field2;
		cudaStatus=cudaMemcpy(d_field1,d_field2,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaCheck(cudaStatus);
			
			
			
			if(step+1<dimW[0])
			{
			add_source<<<gridDim,blockDim>>>(d_wave_propagate_t,d_field2,source_grid,wavelet[step+1],NZ/*dimMod[0]*/,NX/*dimMod[1]*/);
				
			}
			else
			{
			cudaStatus=cudaMemcpy(d_field2,d_wave_propagate_t,NX*NZ/*dimMod[0]*dimMod[1]*/*sizeof(float),cudaMemcpyDeviceToDevice);
				cudaCheck(cudaStatus);			
			}
			
			if(remainderf((float)step,1)==0)
			{
				extractCorrectRegion<<<gridDim,blockDim>>>(d_correctField,d_field2,dimMod[0],dimMod[1],pad);
				cudaStatus=cudaMemcpy(check,d_correctField,dimMod[0]*dimMod[1]*sizeof(float),cudaMemcpyDeviceToHost);
				cudaCheck(cudaStatus);
			// snprintf(buffer, sizeof(char) * 32, "file%d.txt", step);
			// VectoFileWrite(check, size, buffer);
				myFile.write ((char*)check, size*sizeof(float));
			}
			
		}

		
		myFile.close();
		t4=clock();

		printf("Total Execution Time:%f\n",((float)t4-(float)t1)/CLOCKS_PER_SEC);
		printf("Cuda Device Query Execution Time:%f\n",((float)t2-(float)t1)/CLOCKS_PER_SEC);
		printf("Cuda Malloc Execution Time:%f\n",((float)t3-(float)t2)/CLOCKS_PER_SEC);
		printf("Cuda Loop Execution Time:%f\n",((float)t4-(float)t3)/CLOCKS_PER_SEC);

		cudaFree(d_field1);
		cudaFree(d_field2);
		cudaFree(d_wavelet);
		cudaFree(source_grid);
		cudaFree(d_velocity);
		cudaFree(d_wave_propagate_t);
		cudaFree(d_laplace_temp);
		cudaFree(d_laplace);
		
		
		
		


		return 0;
		
		
	}
