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
// limitations under the License.#ifndef _oclConvolution
#include <math.h>

#include "oclConvolution.h"

static cl_float sGauss3[3] = { 1.0f/4.0f, 2.0f/4.0f, 1.0f/2.0f };
static cl_float sGauss5[5] = { 1.0f/17.0f, 4.0f/17.0f, 7.0f/17.0f, 4.0f/17.0f, 1.0f/17.0f };
static cl_float sLoG[5] = { 1.0f/17.0f, 4.0f/17.0f, 7.0f/17.0f, 4.0f/17.0f, 1.0f/17.0f };

oclConvolution::oclConvolution(oclContext& iContext)
: oclProgram(iContext, "oclConvolution")
// buffers
, bfGauss3(iContext, "bfGauss3")
, bfGauss5(iContext, "bfGauss5")
, bfLoG(iContext, "bfLoG")
// kernels
, clConvoluteBuffer3D(*this)
{
	bfGauss3.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 3, sGauss3);
	bfGauss5.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 5, sGauss5);
	bfLoG.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 5, sGauss5);

    addSourceFile("memory\\oclConvolution.cl");
}

//
//
//

int oclConvolution::compile()
{
	clConvoluteBuffer3D = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clConvoluteBuffer3D = createKernel("clConvoluteBuffer3D");
	KERNEL_VALIDATE(clConvoluteBuffer3D)
	return 1;
}

//
//
//

int oclConvolution::compute(oclDevice& iDevice, oclBuffer& bfSource, oclBuffer& bfDest, size_t iDim[3], cl_int4 iAxis, oclBuffer& bfFilter)
{
    cl_int lFilterSize = bfFilter.dim(0)/sizeof(cl_float);
	clSetKernelArg(clConvoluteBuffer3D, 0, sizeof(cl_mem), bfSource);
	clSetKernelArg(clConvoluteBuffer3D, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clConvoluteBuffer3D, 2, sizeof(cl_int4),  &iAxis);
	clSetKernelArg(clConvoluteBuffer3D, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clConvoluteBuffer3D, 4, sizeof(cl_int), &lFilterSize);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clConvoluteBuffer3D, 3, NULL, iDim, 0, 0, NULL, clConvoluteBuffer3D.getEvent());
	ENQUEUE_VALIDATE
	return true;
}   