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
#ifndef _oclFluid3Dnext
#define _oclFluid3Dnext

#include <vector>

#include "sort/oclRadixSort.h"

class oclFluid3Dnext : public oclProgram
{
    public: 

        oclFluid3Dnext(oclContext& iContext);
       ~oclFluid3Dnext();

        int compile();
        int compute(oclDevice& iDevice);
        
        void setParticleCount(size_t iSize);
        size_t getParticleCount();

        int setPositionBuffer(oclBuffer* iBuffer);
        int setStateBuffer(oclBuffer* iBuffer);
        int setForceBuffer(oclBuffer* iBuffer);
        int setBorderBuffer(oclBuffer* iBuffer);

        oclBuffer* getForceBuffer();
        oclBuffer* getPositionBuffer();
        oclBuffer* getStateBuffer();
        oclBuffer* getBorderBuffer();

        oclBuffer& getSortedPositionBuffer();
        oclBuffer& getSortedVelocityBuffer();
        oclBuffer& getIndexBuffer();

        oclBuffer& getParamBuffer();

        typedef struct 
        {
            float deltaTime;
            float kernelRadius;
            float density;
            float stiffness0;
            float stiffness1;
            float viscosity0;
            float viscosity1;
            float radius;

            float cellSize;
            int cellCountX;
            int cellCountY;

        } Params;

        Params& getParameters();


        // events
        static char* EVT_INTEGRATE;
        virtual void addEventHandler(srtEvent& iEvent);

        // sizes
        static const size_t cLocalSize;

    protected:

        oclRadixSort mRadixSort;

        // kernels
        oclKernel clHash;
        oclKernel clReorder;
        oclKernel clInitBounds;
        oclKernel clFindBounds;

        oclKernel clInitFluid;

        oclKernel clComputePressure;
        oclKernel clComputeForces;
        oclKernel clIntegrateForce;
        oclKernel clComputeVelocity;

        // buffers
        oclBuffer bfCell;
        oclBuffer bfCellStart; 
        oclBuffer bfCellEnd;
        oclBuffer bfIndex;

        oclBuffer bfPressure;
        oclBuffer bfForce;
        oclBuffer bfSortedState;

        oclBuffer bfParams;

        oclBuffer* bfBorder;
        oclBuffer* bfPosition;
    
        int bindBuffers();

        void deleteBuffer(oclBuffer* iBuffer)
        {
            if (iBuffer->getOwner<oclFluid3Dnext>() == this)
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
