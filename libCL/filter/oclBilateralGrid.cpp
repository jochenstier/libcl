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

#include "oclBilateralGrid.h"

oclBilateralGrid::oclBilateralGrid(oclContext& iContext)
: oclProgram(iContext, "oclBilateralGrid")
// kernels
, clSplit(*this)
, clSlice2D(*this)
, clSlice3D(*this)
{
	addSourceFile("filter\\oclBilateralGrid.cl");

	exportKernel(clSplit);
	exportKernel(clSlice2D);
	exportKernel(clSlice3D);
}

//
//
//

int oclBilateralGrid::compile()
{
	clSplit = 0;
	clSlice2D = 0;
	clSlice3D = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clSplit = createKernel("clSplit");
	KERNEL_VALIDATE(clSplit)
	clSlice2D = createKernel("clSlice2D");
	KERNEL_VALIDATE(clSlice2D)
	clSlice3D = createKernel("clSlice3D");
	KERNEL_VALIDATE(clSlice3D)
	return 1;
}

//
//
//

int oclBilateralGrid::split(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclImage3D& bfGrid)
{
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clSplit.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clSplit.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

	cl_uint lImageWidth = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	clSetKernelArg(clSplit, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clSplit, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clSplit, 2, sizeof(cl_mem), &iRadius);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clSplit, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clSplit.getEvent());
	ENQUEUE_VALIDATE

	return true;
}


int oclBilateralGrid::slice(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, oclImage2D& bfLine, cl_float4 iMask)
{
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clSlice2D.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clSlice2D.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

	cl_uint lImageWidth = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	lGlobalSize[0] = ceil((float)lImageWidth/lLocalSize[0])*lLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/lLocalSize[1])*lLocalSize[1];

	clSetKernelArg(clSlice2D, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clSlice2D, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clSlice2D, 2, sizeof(cl_uint), &iRadius);
	clSetKernelArg(clSlice2D, 3, sizeof(cl_float), &iRange);
	clSetKernelArg(clSlice2D, 4, sizeof(cl_mem), bfLine);
	clSetKernelArg(clSlice2D, 5, sizeof(cl_float4), &iMask);
 	clSetKernelArg(clSlice2D, 6, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clSlice2D, 7, sizeof(cl_uint), &lImageHeight);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clSlice2D, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clSlice2D.getEvent());
	ENQUEUE_VALIDATE

	return true;
}


int oclBilateralGrid::slice3D(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclImage3D& bfGrid)
{
	size_t lGlobalSize[2];
	lGlobalSize[0] = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);;
	lGlobalSize[1] = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);;
	clSetKernelArg(clSlice3D, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clSlice3D, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clSlice3D, 2, sizeof(cl_mem), bfGrid);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clSlice3D, 2, NULL, lGlobalSize, NULL, 0, NULL, clSlice3D.getEvent());
	ENQUEUE_VALIDATE
	return true;
}

	local clSlice1D = event:getKernel("clSlice1D")
	cl.clSetKernelArg(clSlice1D, 0, bfSrce)
	cl.clSetKernelArg(clSlice1D, 1, bfDest)
	cl.clSetKernelArg(clSlice1D, 2, grid1D)
	cl.clSetKernelArg(clSlice1D, 3, cl.cl_int, _G.GRIDW)
	cl.clSetKernelArg(clSlice1D, 4, cl.cl_int, _G.GRIDH)
	cl.clSetKernelArg(clSlice1D, 5, cl.cl_int, _G.GRIDZ)
	cl.clSetKernelArg(clSlice1D, 6, cl.cl_int, localSize[1])
	cl.clSetKernelArg(clSlice1D, 7, cl.cl_int, localSize[2])
	cl.clEnqueueNDRangeKernel(clSlice1D, 2, { _G.IMAGEW, _G.IMAGEH })
