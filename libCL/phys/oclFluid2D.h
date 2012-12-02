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
#ifndef _oclFluid2D
#define _oclFluid2D

#include <vector>

#include "sort\\oclRadixSort.h"
#include "oclImage2D.h"

class oclFluid2D : public oclProgram
{
    public: 

	    oclFluid2D(oclContext& iContext);
       ~oclFluid2D();

		int compile(); 
        int compute(oclDevice& iDevice);
		
		void setParticleCount(size_t iSize);
		size_t getParticleCount();

		int setStateBuffer(oclBuffer* iBuffer);
		oclBuffer* getStateBuffer();

		void computeBorder(oclImage2D* iBuffer);

		oclBuffer& getParamBuffer();

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

	        float p0; 
	        float vmax;
	        float B;

        } Params;

        typedef struct 
        {
 	        cl_float2 pos;
	        cl_float2 vel;
	        float mass;
	        float unused;

        } Particle;

		Params& getParameters();


		// events
		static char* EVT_INTEGRATE;
        virtual void addEventHandler(srtEvent& iEvent);

        // sizes
		static const size_t cLocalSize = 256;

	protected:

		oclRadixSort mRadixSort;

		// kernels
		oclKernel clHash;
		oclKernel clReorder;
		oclKernel clInitBounds;
		oclKernel clFindBounds;

		oclKernel clInitFluid;

		oclKernel clAdvanceState;
		oclKernel clComputePressure;
		oclKernel clComputePosition;
		oclKernel clUpdateState;

    	oclKernel clComputeBorder;

		// buffers
		oclBuffer bfCell;
		oclBuffer bfCellStart; 
		oclBuffer bfCellEnd;
		oclBuffer bfIndex;

		oclBuffer bfPressure;
		oclBuffer bfRelaxedPos;
		oclBuffer bfPreviousPos;
		oclBuffer* bfState;
		oclBuffer bfSortedState;

		oclBuffer bfParams;

		oclImage2D bfBorderState;
		oclImage2D bfBorderVector;
	
		int bindBuffers();

		void deleteBuffer(oclMem* iBuffer)
		{
			if (iBuffer->getOwner<oclFluid2D>() == this)
			{
				delete iBuffer;
			}
		}

		Params mParams;

		size_t mParticleCount;
	    size_t mCellCount;


    private:


		srtEvent* mIntegrateCb;

};      



#endif
