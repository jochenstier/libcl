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
#ifndef _oclConvolution
#define _oclConvolution

#include "oclProgram.h"
#include "oclBuffer.h"

class oclConvolution : public oclProgram
{
    public: 

	    oclConvolution(oclContext& iContext);

		int compile();
        
        // apply separable convolution over iAxis to 3D buffer
		int compute(oclDevice& iDevice, oclBuffer& bfSource, oclBuffer& bfDest, size_t iDim[3], cl_int4 iAxis, oclBuffer& bfFilter);

        // apply 2D kernel to 3D buffer
		//int compute(oclDevice& iDevice, oclBuffer& bfSource, oclBuffer& bfDest, size_t iDim[3], oclImage2D& bfFilter);

    protected:

 		oclKernel clConvoluteBuffer3D;

        oclBuffer bfGauss3;
        oclBuffer bfGauss5;
        oclBuffer bfLoG;
};      

#endif