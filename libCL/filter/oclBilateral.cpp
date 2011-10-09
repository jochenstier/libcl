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
// limitations under the License.#ifndef _oclBilateral
#include <math.h>

#include "oclBilateral.h"

oclBilateral::oclBilateral(oclContext& iContext)
: oclProgram(iContext, "oclBilateral")
// kernels
, clBilateral(*this)
{
	addSourceFile("filter\\oclBilateral.cl");

	exportKernel(clBilateral);
}

//
//
//

int oclBilateral::compile()
{
	clBilateral = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clBilateral = createKernel("clBilateral");
	KERNEL_VALIDATE(clBilateral)
	return 1;
}

//
//
//

int oclBilateral::compute(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float4 iScalar)
{
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clBilateral.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clBilateral.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

	cl_uint lImageWidth = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	lGlobalSize[0] = ceil((float)lImageWidth/lLocalSize[0])*lLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/lLocalSize[1])*lLocalSize[1];

	clSetKernelArg(clBilateral, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clBilateral, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clBilateral, 2, sizeof(cl_uint), &iRadius);
	clSetKernelArg(clBilateral, 3, sizeof(cl_float4), &iScalar);
 	clSetKernelArg(clBilateral, 4, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clBilateral, 5, sizeof(cl_uint), &lImageHeight);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clBilateral, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clBilateral.getEvent());
	ENQUEUE_VALIDATE

	return true;
}

