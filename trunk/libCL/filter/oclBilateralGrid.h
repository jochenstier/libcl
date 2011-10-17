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

class oclBilateralGrid : public oclProgram
{
    public: 

	    oclBilateralGrid(oclContext& iContext);

		int compile();

		int split(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, cl_float4 iMask);
		int slice2D(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, oclImage2D& bfLine, cl_float4 iMask);
		int slice3D(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclImage3D& bfGrid);

    protected:

		oclKernel clSplit;
		oclKernel clSlice2D;
		oclKernel clSlice3D;
};      

#endif