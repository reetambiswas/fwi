//Shared Global Variables
extern float dt, dx, dz, source_x, bc, source_z;
extern int dimMod[2], dimW[2], NZ, NX;
extern int O; //order of the finite difference
// IN this code, I will put a padding of "pad" which has a cocntinued velocity and 
	// I will solve for this part as well but for saving, I will not save anything from here
extern int pad;
//extern dim3 gridDim, blockDim;

//Block Size of Cuda Kernels
#define Block_Sizex 32
#define Block_Sizey 32

// Defining Constants
#ifndef PI
#define PI 3.141592653589793f
#endif

#ifndef EPS
#define EPS	1e-16
#endif

