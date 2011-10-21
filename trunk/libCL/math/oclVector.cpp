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
#include <math.h>

#include "oclVector.h"

oclVector::oclVector(oclContext& iContext)
: oclProgram(iContext, "oclVector")
// kernels
, clNormalize(*this)
, clXor(*this)
{
    addSourceFile("math\\oclVector.cl");

    exportKernel(clNormalize);
    exportKernel(clXor);
}

//
//
//

int oclVector::compile()
{
	clNormalize = 0;
	clXor = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clNormalize = createKernel("clNormalize");
	KERNEL_VALIDATE(clNormalize)
	clXor = createKernel("clXor");
	KERNEL_VALIDATE(clXor)
	return 1;
}


int oclVector::normalize(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfSrce.dim(0);
    lGlobalSize[1] = bfSrce.dim(1);
	clSetKernelArg(clNormalize, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clNormalize, 1, sizeof(cl_mem), bfDest);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clNormalize, 2, NULL, lGlobalSize, 0, 0, NULL, clNormalize.getEvent());
	ENQUEUE_VALIDATE
    return 1;
};

int oclVector::xor(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_float4 iMask)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfSrce.dim(0);
    lGlobalSize[1] = bfSrce.dim(1);
	clSetKernelArg(clXor, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clXor, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clXor, 2, sizeof(cl_float4),  &iMask);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clXor, 2, NULL, lGlobalSize, 0, 0, NULL, clXor.getEvent());
	ENQUEUE_VALIDATE
    return 1;
};
