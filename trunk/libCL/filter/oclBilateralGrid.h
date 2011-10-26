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
#ifndef _oclBilateralGrid
#define _oclBilateralGrid

#include "oclProgram.h"
#include "oclBuffer.h"
#include "oclImage2D.h"
#include "oclImage3D.h"

#include "util/oclMemory.h"

class oclBilateralGrid : public oclProgram
{
    public: 

	    oclBilateralGrid(oclContext& iContext);

		int compile();

		int split(oclDevice& iDevice, oclImage2D& bfSrce, cl_float4 iMask);
		int slice(oclDevice& iDevice, oclImage2D& bfSrce, cl_float4 iMask, oclImage2D& bfDest);
		int equalize(oclDevice& iDevice, cl_float4 iMask);

		int smoothZ(oclDevice& iDevice, oclBuffer& iKernel);
		int smoothXY(oclDevice& iDevice, oclBuffer& iKernel);
		int smoothXYZ(oclDevice& iDevice, oclBuffer& iKernel);

		void resize(cl_uint iGridW, cl_uint iGridH, cl_uint iGridD);

    protected:

        static cl_int4 sAxisX;
        static cl_int4 sAxisY;
        static cl_int4 sAxisZ;
        oclMemory mMemory;

    	size_t mGridSize[3];

		oclKernel clSplit;
		oclKernel clSlice;
		oclKernel clEqualize;
		oclKernel clConvolute;

        oclImage3D bfGrid3D;
        oclBuffer bfGrid1Da;
        oclBuffer bfGrid1Db;

        oclBuffer* bfCurr;
        oclBuffer* bfTemp;
};      

#endif