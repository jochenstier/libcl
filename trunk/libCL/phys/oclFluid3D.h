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
// limitations under the License.#ifndef _oclFluid3D
#ifndef _oclFluid3D
#define _oclFluid3D

#include <vector>

#include "sort\\oclRadixSort.h"

class oclFluid3D : public oclProgram
{
    public: 

	    oclFluid3D(oclContext& iContext);
       ~oclFluid3D();

		int compile();
        int compute(oclDevice& iDevice);
		
		void setParticleCount(size_t iSize);
		size_t getParticleCount();

		int setPositionBuffer(oclBuffer* iBuffer);
		int setVelocityBuffer(oclBuffer* iBuffer);
		int setForceBuffer(oclBuffer* iBuffer);

		oclBuffer* getForceBuffer();
		oclBuffer* getPositionBuffer();
		oclBuffer* getVelocityBuffer();

		oclBuffer& getSortedPositionBuffer();
		oclBuffer& getSortedVelocityBuffer();
		oclBuffer& getIndexBuffer();

		oclBuffer& getParamBuffer();

		typedef struct 
		{
			float deltaTime;
			float particleRadius;
			float cellSize;
			float mass;
			float viscosity;
			float pressure;
			float density;
			float spacing;
			float stiffness;
			float viscosityConstant;
			float pressureConstant;
			float kernelConstant;
			float velocitylimit;
		} Params;

		Params& getParameters();


		// events
		static char* EVT_INTEGRATE;
        virtual void addEventHandler(srtEvent& iEvent);

        // sizes
		static const size_t cLocalSize = 256;
		static const size_t cBucketCount = 16777216;

	protected:

		oclRadixSort mRadixSort;

		oclKernel clIntegrateForce;
		oclKernel clIntegrateVelocity;
		oclKernel clHash;
		oclKernel clReorder;
		oclKernel clInitBounds;
		oclKernel clFindBounds;
		oclKernel clCalculateDensity;
		oclKernel clCalculateForces;
		oclKernel clInitFluid;
		oclKernel clGravity;
		oclKernel clClipBox;

		oclBuffer bfCell;
		oclBuffer bfCellStart;
		oclBuffer bfCellEnd;
		oclBuffer bfIndex;
		oclBuffer bfSortedPosition;
		oclBuffer bfSortedVelocity;
		oclBuffer bfParams;

		oclBuffer* bfPosition;
		oclBuffer* bfVelocity;
		oclBuffer* bfForce;
	
		int bindBuffers();

		void deleteBuffer(oclBuffer* iBuffer)
		{
			if (iBuffer->getOwner<oclFluid3D>() == this)
			{
				delete iBuffer;
			}
		}

		Params mParams;

		size_t mParticleCount;

    private:


		srtEvent* mIntegrateCb;

};      



#endif
