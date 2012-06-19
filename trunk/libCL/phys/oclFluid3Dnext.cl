// Copyright [2011] [Geist Software Labs Inc.]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

typedef struct 
{
 	float deltaTime;
	float particleRadius;
	float density;
	float stiffness0;
	float stiffness1;
	float viscosity0;
	float viscosity1;
	float radius;

	float cellSize;
	int cellCountX;
	int cellCountY;
	
} System;


//
// map particles into voxel grid
//
__kernel void clHash(__global uint* cell, __global uint* index, __global float4* state, __global const System* param)
{
    const uint particle = get_global_id(0);
    int2 grid;
    grid.x = (int)floor(state[particle].x/param->cellSize);
    grid.y = (int)floor(state[particle].y/param->cellSize);
    cell[particle] = grid.y*param->cellCountX + grid.x;
    index[particle] = particle;
}

__kernel void clReorder(__global const uint *index, __global const float4* stateIn, __global float4* stateOut)
{
    const uint particle= get_global_id(0);
	uint sortedIndex = index[particle];
	stateOut[particle] = stateIn[sortedIndex];
}

__kernel void clInitBounds(__global uint* cellStart, __global uint* cellEnd)
{
    const uint index = get_global_id(0);
	cellStart[index] = 0xFFFFFFFFU;
	cellEnd[index] = 0xFFFFFFFFU;
}

__kernel void clFindBounds(__global uint* cellStart, __global uint* cellEnd, __global const uint* cell, __local uint *localHash)
{
	const uint numParticles = get_global_size(0);
    const uint particle = get_global_id(0);
    const uint threadid = get_local_id(0);

	uint hash = cell[particle];

	localHash[threadid + 1] = hash;
	if(particle > 0 && threadid == 0)
	{
		localHash[0] = cell[particle - 1];
	}
    barrier(CLK_LOCAL_MEM_FENCE);

	if(particle == 0)
	{
		cellStart[hash] = 0;
		return;
	}
	if(hash != localHash[threadid])
	{
		cellEnd[localHash[threadid]]  = cellStart[hash] = particle;
	}
	if(particle == numParticles - 1)
	{
		cellEnd[hash] = numParticles;
	}
}


//
// Compute smoothed particle hydrodynamics 
//

__kernel void clInitFluid(__global System* param)
{
	param->deltaTime = 0.0333333333;

	param->density = 10000.00;
	param->stiffness0 =  0.004;
	param->stiffness1 =  0.01;
	param->viscosity0 =  1;
	param->viscosity1 =  1;

	float spacing = 1;//0.2516;
	param->radius = 12;   

	param->cellSize = 2;   
	param->cellCountX = 1024/param->cellSize;   
	param->cellCountY = 768/param->cellSize;   
	//param->particleRadius = 0.87*pow(param->mass/param->density, 1/3.0f);
	param->particleRadius = 0.87*pow(1.0/param->density, 1/3.0f);
}


float W(float h, float q)
{
	float value = 0;

	q/=h;

	if (q < 1)
	{
		value = 1.0f-3.0f/2.0f*q*q+3.0f/4.0f*q*q*q;
	}
	else if (q < 2.0f)
	{
		q = 2.0f - q;
		value = 1.0f/4.0f*q*q*q;
	}

	return 2.0f/(3.0f*h)*value;
}

__kernel void clComputePressure(__global float4* state, __global float2* pressure, __global const uint* cellStart, __global const uint* cellEnd, __global const System* param)
{
    uint particle = get_global_id(0);
    float4 stateI = state[particle];

    int2 grid;
    grid.x = (int)floor(stateI.x/param->cellSize);
    grid.y = (int)floor(stateI.y/param->cellSize);

	int x0=max(grid.x-1,0);
	int x1=min(grid.x+1,param->cellCountX-1);
	int y0=max(grid.y-1,0);
	int y1=min(grid.y+1,param->cellCountY-1);

    float density = 0;
	for(int x = x0; x <= x1; x++)
	{
		for(int y = y0; y <= y1; y++)
		{
			int cell = y*param->cellCountX + x;

			uint start = cellStart[cell];
			if (start != 0xFFFFFFFFU)
			{
				uint end = cellEnd[cell];
				for(uint j = start; j < end; j++)
				{
					if(j != particle)
					{
						float4 stateJ = state[j];
						float2 rIJ = stateJ.xy - stateI.xy;
						density += W(param->cellSize, length(rIJ));
					}
				}
			}
		}
	}
	if (density != 0)
	{
		float tait = 0.87*(pow(density/100.0f, 7.0f)-1.0f);
		pressure[particle] = tait/(density*density);
	}
	else 
	{
		pressure[particle] = 0;
	}
}

__kernel void clComputeForces(__global float4* state, __global float2* pressure, __global float2* force, __global const uint* cellStart, __global const uint* cellEnd, __global const System* param)
{
    uint particle = get_global_id(0);
    float2 pressureI = pressure[particle];
    float4 stateI = state[particle];

    int2 grid;
    grid.x = (int)floor(stateI.x/param->cellSize);
    grid.y = (int)floor(stateI.y/param->cellSize);

	int x0=max(grid.x-1,0);
	int x1=min(grid.x+1,param->cellCountX-1);
	int y0=max(grid.y-1,0);
	int y1=min(grid.y+1,param->cellCountY-1);

	float2 Fviscosity = 0;
	float2 Fpressure = 0;
	float2 Fsurface = 0;

    float2 dsp = 0;
	for(int x = x0; x <= x1; x++)
	{
		for(int y = y0; y <= y1; y++)
		{
			int cell = y*param->cellCountX + x;

			uint start = cellStart[cell];
			if (start != 0xFFFFFFFFU)
			{
				uint end = cellEnd[cell];
				for(uint j = start; j < end; j++)
				{
					if(j != particle)
					{
						float4 stateJ = state[j];

						float2 rIJ = stateJ.xy - stateI.xy;
						float r = length(rIJ);
						float2 w = W(param->cellSize, r);
						float2 g = rIJ/r*w;

						float2 pressureJ = pressure[j];

						Fpressure = -(pressureI.x + pressureJ.x)*g;
						Fsurface = rIJ*w;
						//Fviscosity +=
						//Ftension +=

					}
				}
			}
		}
	}

	force[particle] = (Fpressure + Fviscosity + 0.25*Fsurface) - (float2)(0,9.81);
}


__kernel void clComputeVelocity(__global float4* currPos, __global float4* currVel, __global uint* index, __global float4* nextPos, __global float4* dspPos,  __global const System* param)
{
    const uint particle = get_global_id(0);
    float4 pos0 = currPos[index[particle]];
    float4 pos1 = nextPos[particle] + dspPos[particle];

	currVel[index[particle]] = (pos1 - pos0)/param->deltaTime;
	currPos[index[particle]] = pos1;
}

__kernel void clIntegrateForce(__global float4* state, __global float2* force, __global uint* index, __global const System* param)
{
    const uint particle = get_global_id(0);
	uint sorted = index[particle];

	float4 currState = state[sorted];	
	currState.zw += force[particle]*param->deltaTime;

	state[sorted].xy += currState.zw*param->deltaTime;
	state[sorted].xy = clamp(state[sorted].xy, (float2)(1,1), (float2)(1024,768));

	state[sorted].zw = currState.zw;
}
