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

#include "oclMemory.h"


cl_float4 oclMemory::c0000 = { 0, 0, 0, 0 };


oclMemory::oclMemory(oclContext& iContext)
: oclProgram(iContext, "oclMemory")
// kernels
, clMemSet(*this)
{
    addSourceFile("util\\oclMemory.cl");

    exportKernel(clMemSet);
}

//
//
//

int oclMemory::compile()
{
	clMemSet = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clMemSet = createKernel("clMemSet");
	KERNEL_VALIDATE(clMemSet)
     
	return 1;
}


int oclMemory::memSet(oclDevice& iDevice, oclImage2D& bfDest, cl_float4 iValue)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfDest.dim(0);
    lGlobalSize[1] = bfDest.dim(1);
	clSetKernelArg(clMemSet, 0, sizeof(cl_mem), bfDest);
	clSetKernelArg(clMemSet, 1, sizeof(cl_float4), &iValue);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clMemSet, 2, NULL, lGlobalSize, 0, 0, NULL, clMemSet.getEvent());
	ENQUEUE_VALIDATE
    return 1;
};
