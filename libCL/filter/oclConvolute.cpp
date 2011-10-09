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
// limitations under the License.#ifndef _oclConvolute
#include <math.h>

#include "oclConvolute.h"

static cl_float sGauss3[3] = { 1.0f/4.0f, 2.0f/4.0f, 1.0f/2.0f };
static cl_float sGauss5[5] = { 1.0f/17.0f, 4.0f/17.0f, 7.0f/17.0f, 4.0f/17.0f, 1.0f/17.0f };
static cl_float sLoG[5] = { 1.0f/17.0f, 4.0f/17.0f, 7.0f/17.0f, 4.0f/17.0f, 1.0f/17.0f };

oclConvolute::oclConvolute(oclContext& iContext)
: oclProgram(iContext, "oclConvolute")
// buffers
, bfGauss3(iContext, "bfGauss3")
, bfGauss5(iContext, "bfGauss5")
, bfLoG(iContext, "bfLoG")
// kernels
, clConv3D(*this)
, clConv2D(*this)
{
	bfGauss3.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 3, sGauss3);
	bfGauss5.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 5, sGauss5);
	bfLoG.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 5, sGauss5);

    addSourceFile("filter\\oclConvolute.cl");
}

//
//
//

int oclConvolute::compile()
{
	clConv3D = 0;
	clConv2D = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clConv3D = createKernel("clConv3D");
	KERNEL_VALIDATE(clConv3D)
	clConv2D = createKernel("clConv2D");
	KERNEL_VALIDATE(clConv2D)
	return 1;
}

//
//
//

int oclConvolute::conv3D(oclDevice& iDevice, oclBuffer& bfSource, oclBuffer& bfDest, size_t iDim[3], cl_int4 iAxis, oclBuffer& bfFilter)
{
    cl_int lFilterSize = bfFilter.dim(0)/sizeof(cl_float);
	clSetKernelArg(clConv3D, 0, sizeof(cl_mem), bfSource);
	clSetKernelArg(clConv3D, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clConv3D, 2, sizeof(cl_int4),  &iAxis);
	clSetKernelArg(clConv3D, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clConv3D, 4, sizeof(cl_int), &lFilterSize);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clConv3D, 3, NULL, iDim, 0, 0, NULL, clConv3D.getEvent());
	ENQUEUE_VALIDATE
	return true;
}   


int oclConvolute::conv2D(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int4 iAxis, oclBuffer& bfFilter)
{
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clConv2D.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clConv2D.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

	cl_uint lImageWidth = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	lGlobalSize[0] = ceil((float)lImageWidth/lLocalSize[0])*lLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/lLocalSize[1])*lLocalSize[1];

    cl_int lFilter2DSize = bfFilter.dim(0)/sizeof(cl_float);
    if (lFilter2DSize %2 == 0)
    {
        Log(ERR, this) << "Failure in call to oclFilter2D::conv1 : convolution size must be odd ";
    }
	clSetKernelArg(clConv2D, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clConv2D, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clConv2D, 2, sizeof(cl_int4),  &iAxis);
	clSetKernelArg(clConv2D, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clConv2D, 4, sizeof(cl_int), &lFilter2DSize);
 	clSetKernelArg(clConv2D, 5, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clConv2D, 6, sizeof(cl_uint), &lImageHeight);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clConv2D, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clConv2D.getEvent());
	ENQUEUE_VALIDATE
	return true;
}   
