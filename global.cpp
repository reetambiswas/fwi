#include"global.h"
#include"header.h"


float dt, dx, dz, source_x, bc, source_z;
int dimMod[2], dimW[2], NX, NZ;

int O; //order of the finite difference
// IN this code, I will put a padding of "pad" which has a cocntinued velocity and 
	// I will solve for this part as well but for saving, I will not save anything from here
int pad;