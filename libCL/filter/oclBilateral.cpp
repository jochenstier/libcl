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

#include "oclBilateral.h"

oclBilateral::oclBilateral(oclContext& iContext)
: oclProgram(iContext, "oclBilateral")
// kernels
, clIso2D(*this)
, clAniso2Dtang(*this)
, clAniso2Dorth(*this)
{
	addSourceFile("filter/oclBilateral.cl");

	exportKernel(clIso2D);
	exportKernel(clAniso2Dtang);
	exportKernel(clAniso2Dorth);
}

//
//
//

int oclBilateral::compile()
{
	clIso2D = 0;
	clAniso2Dtang = 0;
	clAniso2Dorth = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clIso2D = createKernel("clIso2D");
	KERNEL_VALIDATE(clIso2D)
	clAniso2Dtang = createKernel("clAniso2Dtang");
	KERNEL_VALIDATE(clAniso2Dtang)
	clAniso2Dorth = createKernel("clAniso2Dorth");
	KERNEL_VALIDATE(clAniso2Dorth)
	return 1;
}

//
//
//

int oclBilateral::iso2D(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, cl_float4 iMask)
{
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clIso2D.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

	clSetKernelArg(clIso2D, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clIso2D, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clIso2D, 2, sizeof(cl_uint), &iRadius);
	clSetKernelArg(clIso2D, 3, sizeof(cl_float), &iRange);
	clSetKernelArg(clIso2D, 4, sizeof(cl_float4), &iMask);
 	clSetKernelArg(clIso2D, 5, sizeof(cl_uint), &lIw);
	clSetKernelArg(clIso2D, 6, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clIso2D, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clIso2D.getEvent());
	ENQUEUE_VALIDATE

	return true;
}


int oclBilateral::aniso2Dtang(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, oclImage2D& bfLine, cl_float4 iMask)
{
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clAniso2Dtang.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

	clSetKernelArg(clAniso2Dtang, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clAniso2Dtang, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clAniso2Dtang, 2, sizeof(cl_uint), &iRadius);
	clSetKernelArg(clAniso2Dtang, 3, sizeof(cl_float), &iRange);
	clSetKernelArg(clAniso2Dtang, 4, sizeof(cl_mem), bfLine);
	clSetKernelArg(clAniso2Dtang, 5, sizeof(cl_float4), &iMask);
 	clSetKernelArg(clAniso2Dtang, 6, sizeof(cl_uint), &lIw);
	clSetKernelArg(clAniso2Dtang, 7, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clAniso2Dtang, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clAniso2Dtang.getEvent());
	ENQUEUE_VALIDATE

	return true;
}


int oclBilateral::aniso2Dorth(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, oclImage2D& bfLine, cl_float4 iMask)
{
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clAniso2Dorth.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

	clSetKernelArg(clAniso2Dorth, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clAniso2Dorth, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clAniso2Dorth, 2, sizeof(cl_uint), &iRadius);
	clSetKernelArg(clAniso2Dorth, 3, sizeof(cl_float), &iRange);
	clSetKernelArg(clAniso2Dorth, 4, sizeof(cl_mem), bfLine);
	clSetKernelArg(clAniso2Dorth, 5, sizeof(cl_float4), &iMask);
 	clSetKernelArg(clAniso2Dorth, 6, sizeof(cl_uint), &lIw);
	clSetKernelArg(clAniso2Dorth, 7, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clAniso2Dorth, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clAniso2Dorth.getEvent());
	ENQUEUE_VALIDATE

	return true;
}

