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
#ifndef _oclBloom
#define _oclBloom

#include "oclProgram.h"
#include "oclBuffer.h"
#include "oclKernel.h"
#include "oclContext.h"

#include "memory\\oclRecursiveGaussian.h"

class oclBloom : public oclProgram
{
    public: 

	    oclBloom(oclContext& iContext, cl_image_format iFormat = sDefaultFormat);

		int compile();
        int compute(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest);

        void setSmoothing(cl_float iValue);
        void setThreshold(cl_float iValue);
        void setIntensity(cl_float iValue);

    protected:

		oclRecursiveGaussian mGaussian;

		oclKernel clFilter;
		oclKernel clCombine;

		oclImage2D bfTempA;
		oclImage2D bfTempB;

		static cl_image_format sDefaultFormat;
};      

#endif