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
#ifndef _oclFilter2D
#define _oclFilter2D

#include "oclProgram.h"
#include "oclBuffer.h"
#include "oclImage2D.h"

class oclFilter2D : public oclProgram
{
    public: 

	    oclFilter2D(oclContext& iContext);

		int compile();

		int conv1(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int4 iAxis, oclBuffer& bfFilter2D);
		int conv2(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclBuffer& bfFilter2D);

		int bilateral(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float4 iRange);
		int sobel(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDx, oclImage2D& bfDy);
		int tangent(oclDevice& iDevice, oclImage2D& bfDx, oclImage2D& bfDy, oclImage2D& bfDest);

    protected:

        oclBuffer bfGauss3;
        oclBuffer bfGauss5;

		oclKernel clBilateral;
		oclKernel clSobel;
		oclKernel clTangent;
        
        oclKernel clConv1;
        oclKernel clConv2;
};      

#endif