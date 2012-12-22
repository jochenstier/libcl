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
 	float dT;

	float h;
	float containerW;
	float containerH;
	float cellSize;
	int cellCountX;
	int cellCountY;

	float particleSize;
	float particleMass;

	float restDensity;
	float vmax;
	float B;

} System;

typedef struct
{
	float2 pos;
	float2 vel;
	float mass;
	float age;

} Particle;

typedef union
{
	struct
	{
		float density;
		float nearDensity;
		float pressure;
		float nearPressure;
	};

	float4 s;

} State;

//
// GRID
//

__kernel void clHash(__global uint* cell, __global uint* index, __global Particle* state, __global const System* param)
{
    const uint particle = get_global_id(0);
    int2 grid;
    grid.x = (int)floor(state[particle].pos.x/param->cellSize);
    grid.y = (int)floor(state[particle].pos.y/param->cellSize);
    cell[particle] = grid.y*param->cellCountX + grid.x;
    index[particle] = particle;
}

__kernel void clReorder(__global const uint *index, __global Particle* stateIn, __global Particle* stateOut)
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

#define kRestDensity 82.0f
#define kStiffness 0.08f
#define kNearStiffness 0.1f
#define kSurfaceTension 21.00035f  // 0.0004f
#define kLinearViscocity 0.5f
#define kQuadraticViscocity 1.0f
#define kPi 3.1415926535f

#define kParticleRadius 0.05f/5   // 0.05f
#define kH (22*kParticleRadius)  // 6*kParticleRadius

#define kNorm (20/(2*kPi*kH*kH))
#define kNearNorm (30/(2*kPi*kH*kH))


__kernel void clInitFluid(__global System* param)
{
	param->dT = 0.00433333333;  //0.00133333333;

	param->h = kH;  // m
	param->containerW = 17.68; // m
	param->containerH = 15.76; // m
	param->cellSize = 2*param->h; 
	param->cellCountX = param->containerW/param->cellSize+0.5;
	param->cellCountY = param->containerH/param->cellSize+0.5;   

	param->particleSize = kParticleRadius;   
	param->particleMass = 0.7f;//0.7f;  

	param->restDensity = kRestDensity;  
}

__kernel void clInitParticles(__global Particle* particle)
{
    const uint index = get_global_id(0);

	Particle stateI;
	stateI.vel = (float2)(0,0);
	stateI.pos = (float2)(0,0);
	stateI.mass = 0.7;
	stateI.age = 0;
	particle[index] = stateI;
}

//
// SPH
//


#define EPSILON 0.0000001f


__kernel void clAdvanceState(__global Particle* particle, __global float2* previousPosition, __global const System* param)
{
    const uint index = get_global_id(0);

	Particle stateI = particle[index];
	previousPosition[index] =  stateI.pos;
	stateI.vel += (float2)(0,-9.81)*param->dT;
	stateI.pos += stateI.vel*param->dT;
	particle[index] = stateI;
}


__kernel void clComputePressure(__global Particle* particle, __global State* state, __global const uint* cellStart, __global const uint* cellEnd, __read_only image2d_t borderState, __global const System* param)
{
    uint index = get_global_id(0);
    Particle particleI = particle[index];

    int2 grid;
    grid.x = (int)floor(particleI.pos.x/param->cellSize);
    grid.y = (int)floor(particleI.pos.y/param->cellSize);

	int x0=max(grid.x-1,0);
	int x1=min(grid.x+1,param->cellCountX-1);
	int y0=max(grid.y-1,0);
	int y1=min(grid.y+1,param->cellCountY-1);

	float density = 0;
	float nearDensity = 0;

	State wall;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
	wall.s = read_imagef(borderState, sampler, particleI.pos/(float2)(param->containerW, param->containerH));
	if (wall.density != 0)
	{
		density += wall.density;
		nearDensity += wall.nearDensity;
	}

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
					if(j != index)
					{
						Particle particleJ = particle[j];
						float2 dxy = particleJ.pos - particleI.pos;
						float r = length(dxy);

						if (r > EPSILON && r < param->h)
						{
							float a = 1 - r/param->h;
							density += param->particleMass * a*a*a * kNorm;
							nearDensity += param->particleMass * a*a*a*a * kNearNorm;
						}
					}
				}
			}
		}
	}

	state[index].density = density;
	state[index].nearDensity = nearDensity; 
	state[index].pressure = kStiffness * (density - param->particleMass*kRestDensity);
	state[index].nearPressure = kNearStiffness * nearDensity; 
}

__kernel void clComputePosition(__global Particle* particle, __global State* state, __global float2* relaxedPosition, __global const uint* cellStart, __global const uint* cellEnd, __read_only image2d_t borderState, __read_only image2d_t borderVector, __global const System* param)
{
    uint index = get_global_id(0);
    State stateI = state[index];
    Particle particleI = particle[index];

    int2 grid;
    grid.x = (int)floor(particleI.pos.x/param->cellSize);
    grid.y = (int)floor(particleI.pos.y/param->cellSize);

	int x0=max(grid.x-1,0);
	int x1=min(grid.x+1,param->cellCountX-1);
	int y0=max(grid.y-1,0);
	int y1=min(grid.y+1,param->cellCountY-1);

	float2 rxy = particleI.pos;

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
					if(j != index)
					{
						Particle particleJ = particle[j];
						float2 dxy = particleJ.pos- particleI.pos;
						float r = length(dxy);

						if (r > EPSILON && r < param->h)
						{
							State stateJ = state[j];

							float a = 1 - r/param->h;

							// pressure
							float d = (param->dT*param->dT) * ((stateI.nearPressure + stateJ.nearPressure)*a*a*a*kNearNorm + (stateI.pressure + stateJ.pressure)*a*a*kNorm) / 2;
							rxy -= dxy*d/(r*param->particleMass);

							// surface tension
							if (param->particleMass == param->particleMass)
							{
								rxy += (param->dT*param->dT) * (kSurfaceTension/param->particleMass) * param->particleMass*(a*a*a*kNearNorm+a*a*kNorm )/ 2* dxy;
							}

							// viscocity
							float2 duv = particleJ.vel - particleJ.vel;
							float u = dot(duv, dxy);
							if (u > 0)
							{
								u /= r;
								float I = 0.5f * param->dT * a * (kLinearViscocity*u + kQuadraticViscocity*u*u);
								rxy -= I * dxy * param->dT;
							}
						}
					}
				}
			}
		}
	}

	State wall;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
	wall.s = read_imagef(borderState, sampler, rxy/(float2)(param->containerW, param->containerH));
	if (wall.density != 0)
	{
		float4 normal = read_imagef(borderVector, sampler, rxy/(float2)(param->containerW, param->containerH));
		float a = 1 - normal.z/param->h;
		float d = (param->dT*param->dT) * ((stateI.nearPressure + wall.nearPressure)*a*a*a*kNearNorm + (stateI.pressure+wall.pressure)*a*a*kNorm)/ 2;
		rxy += normal.xy*d/param->particleMass;
	}

	relaxedPosition[index] = rxy;
}


__kernel void clUpdateState(__global Particle* state, __global float2* previousPosition, __global float2* relaxedPosition, __global uint* index, __global const System* param)
{
    const uint particle = get_global_id(0);
	uint sorted = index[particle];

	float2 dxy = relaxedPosition[particle].xy - previousPosition[sorted];
	float l = length(dxy);
	dxy = dxy*min(l, 4.0*param->particleSize)/l;
	state[sorted].vel = dxy/param->dT;
	state[sorted].pos = previousPosition[sorted]+dxy;
}


__kernel void clEmitter(__global Particle* state, uint emit, __global const System* param)
{
    const uint particle = get_global_id(0);
    if (particle > emit && particle < emit+2)
    {
		int index = emit - particle;
		state[particle].pos = (float2)(1,14+(particle%4)*12*param->particleSize);
		state[particle].vel = (float2)(25, 1);
		state[particle].age = 1;
    }
}

//
//
//

__kernel void clComputeBorder(__read_only image2d_t borderIn, __write_only image2d_t vecOut, __write_only image2d_t stateOut, __global const System* param)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

	float4 pixel = read_imagef(borderIn, sampler, (int2)(x,y));

	float dmin=10000;
	int dW = w*param->h/param->containerW + 0.5;
	int dH = h*param->h/param->containerH + 0.5;
	float dX = param->containerW/w;
	float dY = param->containerH/h;


	float4 vector = 0;
	float density = 0;
	float nearDensity = 0;

	if (pixel.w == 1.0f)
	{
		// inside wall
		density = 112*param->particleMass *  kNorm;
		nearDensity = 112*param->particleMass *  kNearNorm;
		for (int i= max(x-dW,0); i<min(x+dW,w); i++)
		{
			for (int j=max(y-dH,0); j<min(y+dH,h); j++)
			{
				float4 neighbour = read_imagef(borderIn, sampler, (int2)(i,j));
				if (neighbour.w != 1.0f)
				{
					float2 dxy = (float2)(i*dX, j*dY) - (float2)(x*dX, y*dY);
					float r = length(dxy);
					if (r < dmin)
					{
						vector.xy = dxy/r;
						vector.z = r;
						dmin = r;
					}
				}
			}
		}
	}
	else
	{
		// outside wall
		for (int i= max(x-dW,0); i<min(x+dW,w); i++)
		{
			for (int j=max(y-dH,0); j<min(y+dH,h); j++)
			{
				float4 neighbour = read_imagef(borderIn, sampler, (int2)(i,j));
				if (neighbour.w == 1.0f)
				{
					float2 dxy = (float2)(x*dX, y*dY) - (float2)(i*dX, j*dY);
					float r = length(dxy);
					if (r < param->h)
					{
						float s = (dX*dY)/(2*M_PI*param->particleSize*param->particleSize);
						float a = 1 - r/param->h;
						density += s*param->particleMass * a*a*a * kNorm;
						nearDensity += s*param->particleMass * a*a*a*a * kNearNorm;

						if (r < dmin)
						{
							vector.xy = dxy/r;
							vector.z = r;
							dmin = r;
						}
					}
				}
			}
		}
	}

	write_imagef(vecOut, (int2)(x,y), (float4)vector);

	float4 state;
	state.x = density;
	state.y = nearDensity; 
	state.z = kStiffness * (density - param->particleMass*kRestDensity);
	state.w = kNearStiffness * nearDensity; 
	write_imagef(stateOut, (int2)(x,y), state);

}


/*
typedef struct 
{
 	float dT;

	float h;
	float containerW;
	float containerH;
	float cellSize;
	int cellCountX;
	int cellCountY;

	float particleSize;
	float particleMass;

	float restDensity;
	float vmax;
	float B;

} System;

typedef struct
{
	float2 pos;
	float2 vel;
	//float2 fff;

} Particle;

typedef union
{
	struct
	{
		float density;
		float nearDensity;
		float pressure;
		float nearPressure;
	};

	float4 s;

} State;

//
// GRID
//

__kernel void clHash(__global uint* cell, __global uint* index, __global Particle* state, __global const System* param)
{
    const uint particle = get_global_id(0);
    int2 grid;
    grid.x = (int)floor(state[particle].pos.x/param->cellSize);
    grid.y = (int)floor(state[particle].pos.y/param->cellSize);
    cell[particle] = grid.y*param->cellCountX + grid.x;
    index[particle] = particle;
}

__kernel void clReorder(__global const uint *index, __global Particle* stateIn, __global Particle* stateOut)
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

#define kRestDensity 82.0f
#define kStiffness 0.08f
#define kNearStiffness 0.1f
#define kSurfaceTension 21.00035f  // 0.0004f
#define kLinearViscocity 0.5f
#define kQuadraticViscocity 1.0f
#define kPi 3.1415926535f

#define kParticleRadius 0.05f/5   // 0.05f
#define kH (22*kParticleRadius)  // 6*kParticleRadius

#define kNorm (20/(2*kPi*kH*kH))
#define kNearNorm (30/(2*kPi*kH*kH))


__kernel void clInitFluid(__global System* param)
{
	param->dT = 0.00433333333;  //0.00133333333;

	param->h = kH;  // m
	param->containerW = 17.68; // m
	param->containerH = 15.76; // m
	param->cellSize = 2*param->h; 
	param->cellCountX = param->containerW/param->cellSize+0.5;
	param->cellCountY = param->containerH/param->cellSize+0.5;   

	param->particleSize = kParticleRadius;   
	param->particleMass = 0.7f;//0.7f;  

	param->restDensity = kRestDensity;  
}

//
// SPH
//


#define EPSILON 0.0000001f


__kernel void clAdvanceState(__global Particle* state, __global float2* previousPosition, __global const System* param)
{
    const uint particle = get_global_id(0);

	Particle stateI = state[particle];
	previousPosition[particle] =  stateI.pos;
	stateI.vel += (float2)(0,-9.81)*param->dT;
	stateI.pos += stateI.vel*param->dT;
	state[particle] = stateI;
}


__kernel void clComputePressure(__global Particle* particle, __global State* state, __global const uint* cellStart, __global const uint* cellEnd, __read_only image2d_t borderState, __global const System* param)
{
    uint index = get_global_id(0);
    Particle particleI = particle[index];

    int2 grid;
    grid.x = (int)floor(particleI.pos.x/param->cellSize);
    grid.y = (int)floor(particleI.pos.y/param->cellSize);

	int x0=max(grid.x-1,0);
	int x1=min(grid.x+1,param->cellCountX-1);
	int y0=max(grid.y-1,0);
	int y1=min(grid.y+1,param->cellCountY-1);

	float density = 0;
	float nearDensity = 0;

	State wall;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
	wall.s = read_imagef(borderState, sampler, particleI.pos/(float2)(param->containerW, param->containerH));
	if (wall.density != 0)
	{
		density += wall.density;
		nearDensity += wall.nearDensity;
	}

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
					if(j != index)
					{
						Particle particleJ = particle[j];
						float2 dxy = particleJ.pos - particleI.pos;
						float r = length(dxy);

						if (r > EPSILON && r < param->h)
						{
							float a = 1 - r/param->h;
							density += param->particleMass * a*a*a * kNorm;
							nearDensity += param->particleMass * a*a*a*a * kNearNorm;
						}
					}
				}
			}
		}
	}

	state[index].density = density;
	state[index].nearDensity = nearDensity; 
	state[index].pressure = kStiffness * (density - param->particleMass*kRestDensity);
	state[index].nearPressure = kNearStiffness * nearDensity; 
}

__kernel void clComputePosition(__global Particle* particle, __global State* state, __global float2* relaxedPosition, __global const uint* cellStart, __global const uint* cellEnd, __read_only image2d_t borderState, __read_only image2d_t borderVector, __global const System* param)
{
    uint index = get_global_id(0);
    State stateI = state[index];
    Particle particleI = particle[index];

    int2 grid;
    grid.x = (int)floor(particleI.pos.x/param->cellSize);
    grid.y = (int)floor(particleI.pos.y/param->cellSize);

	int x0=max(grid.x-1,0);
	int x1=min(grid.x+1,param->cellCountX-1);
	int y0=max(grid.y-1,0);
	int y1=min(grid.y+1,param->cellCountY-1);

	float2 rxy = particleI.pos;

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
					if(j != index)
					{
						Particle particleJ = particle[j];
						float2 dxy = particleJ.pos- particleI.pos;
						float r = length(dxy);

						if (r > EPSILON && r < param->h)
						{
							State stateJ = state[j];

							float a = 1 - r/param->h;

							// pressure
							float d = (param->dT*param->dT) * ((stateI.nearPressure + stateJ.nearPressure)*a*a*a*kNearNorm + (stateI.pressure + stateJ.pressure)*a*a*kNorm) / 2;
							rxy -= dxy*d/(r*param->particleMass);

							// surface tension
							if (param->particleMass == param->particleMass)
							{
								rxy += (param->dT*param->dT) * (kSurfaceTension/param->particleMass) * param->particleMass*(a*a*a*kNearNorm+a*a*kNorm )/ 2* dxy;
							}

							// viscocity
							float2 duv = particleJ.vel - particleJ.vel;
							float u = dot(duv, dxy);
							if (u > 0)
							{
								u /= r;
								float I = 0.5f * param->dT * a * (kLinearViscocity*u + kQuadraticViscocity*u*u);
								rxy -= I * dxy * param->dT;
							}
						}
					}
				}
			}
		}
	}

	State wall;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
	wall.s = read_imagef(borderState, sampler, rxy/(float2)(param->containerW, param->containerH));
	if (wall.density != 0)
	{
		float4 normal = read_imagef(borderVector, sampler, rxy/(float2)(param->containerW, param->containerH));
		float a = 1 - normal.z/param->h;
		float d = (param->dT*param->dT) * ((stateI.nearPressure + wall.nearPressure)*a*a*a*kNearNorm + (stateI.pressure+wall.pressure)*a*a*kNorm)/ 2;
		rxy += normal.xy*d/param->particleMass;
	}

	relaxedPosition[index] = rxy;
}


__kernel void clUpdateState(__global Particle* state, __global float2* previousPosition, __global float2* relaxedPosition, __global uint* index, __global const System* param)
{
    const uint particle = get_global_id(0);
	uint sorted = index[particle];

	float2 dxy = relaxedPosition[particle].xy - previousPosition[sorted];
	float l = length(dxy);
	dxy = dxy*min(l, 2.9*param->particleSize)/l;
	state[sorted].vel = dxy/param->dT;
	state[sorted].pos = previousPosition[sorted]+dxy;
}

//
//
//

__kernel void clComputeBorder(__read_only image2d_t borderIn, __write_only image2d_t vecOut, __write_only image2d_t stateOut, __global const System* param)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

	float4 pixel = read_imagef(borderIn, sampler, (int2)(x,y));

	float dmin=10000;
	int dW = w*param->h/param->containerW + 0.5;
	int dH = h*param->h/param->containerH + 0.5;
	float dX = param->containerW/w;
	float dY = param->containerH/h;


	float4 vector = 0;
	float density = 0;
	float nearDensity = 0;

	if (pixel.w == 1.0f)
	{
		// inside wall
		density = param->particleMass *  kNorm;
		nearDensity = param->particleMass *  kNearNorm;
		for (int i= max(x-dW,0); i<min(x+dW,w); i++)
		{
			for (int j=max(y-dH,0); j<min(y+dH,h); j++)
			{
				float4 neighbour = read_imagef(borderIn, sampler, (int2)(i,j));
				if (neighbour.w != 1.0f)
				{
					float2 dxy = (float2)(i*dX, j*dY) - (float2)(x*dX, y*dY);
					float r = length(dxy);
					if (r < dmin)
					{
						vector.xy = dxy/r;
						vector.z = r;
						dmin = r;
					}
				}
			}
		}
	}
	else
	{
		// outside wall
		for (int i= max(x-dW,0); i<min(x+dW,w); i++)
		{
			for (int j=max(y-dH,0); j<min(y+dH,h); j++)
			{
				float4 neighbour = read_imagef(borderIn, sampler, (int2)(i,j));
				if (neighbour.w == 1.0f)
				{
					float2 dxy = (float2)(x*dX, y*dY) - (float2)(i*dX, j*dY);
					float r = length(dxy);
					if (r < param->h)
					{
						float s = (dX*dY)/(2*M_PI*param->particleSize*param->particleSize);
						float a = 1 - r/param->h;
						density += s*param->particleMass * a*a*a * kNorm;
						nearDensity += s*param->particleMass * a*a*a*a * kNearNorm;

						if (r < dmin)
						{
							vector.xy = dxy/r;
							vector.z = r;
							dmin = r;
						}
					}
				}
			}
		}
	}

	write_imagef(vecOut, (int2)(x,y), (float4)vector);

	float4 state;
	state.x = density;
	state.y = nearDensity; 
	state.z = kStiffness * (density - param->particleMass*kRestDensity);
	state.w = kNearStiffness * nearDensity; 
	write_imagef(stateOut, (int2)(x,y), state);

}
*/