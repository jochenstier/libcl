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
// limitations under the License.#ifndef _oclFilter2D
#include <math.h>

#include "oclFilter2D.h"

static cl_float sGauss3[3] = { 1.0f/4.0f, 2.0f/4.0f, 1.0f/2.0f };
static cl_float sGauss5[5] = { 1.0f/17.0f, 4.0f/17.0f, 7.0f/17.0f, 4.0f/17.0f, 1.0f/17.0f };

oclFilter2D::oclFilter2D(oclContext& iContext)
: oclProgram(iContext, "oclFilter2D")
// buffers
, bfGauss3(iContext, "bfGauss3")
, bfGauss5(iContext, "bfGauss5")
// kernels
, clBilateral(*this)
, clSobel(*this)
, clTangent(*this)
, clConv1(*this)
, clConv2(*this)
{
	bfGauss3.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 3, sGauss3);
	bfGauss5.create<cl_float>(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 5, sGauss5);

	addSourceFile("image\\oclFilter2D.cl");

	exportKernel(clBilateral);
}

//
//
//

int oclFilter2D::compile()
{
	clBilateral = 0;
	clSobel = 0;
	clTangent = 0;
    clConv1 = 0;
    clConv2 = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clConv1 = createKernel("clConv1");
	KERNEL_VALIDATE(clConv1)
	clConv2 = createKernel("clConv2");
	KERNEL_VALIDATE(clConv2)

	clSobel = createKernel("clSobel");
	KERNEL_VALIDATE(clSobel)
	clBilateral = createKernel("clBilateral");
	KERNEL_VALIDATE(clBilateral)
	clTangent = createKernel("clTangent");
	KERNEL_VALIDATE(clTangent)
	return 1;
}

//
//
//

int oclFilter2D::conv1(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int4 iAxis, oclBuffer& bfFilter)
{
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clConv2.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clConv2.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

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
	clSetKernelArg(clConv1, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clConv1, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clConv1, 2, sizeof(cl_int4),  &iAxis);
	clSetKernelArg(clConv1, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clConv1, 4, sizeof(cl_int), &lFilter2DSize);
 	clSetKernelArg(clConv1, 5, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clConv1, 6, sizeof(cl_uint), &lImageHeight);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clConv1, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clConv1.getEvent());
	ENQUEUE_VALIDATE
	return true;
}   

int oclFilter2D::conv2(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclBuffer& bfFilter)
{
    /*
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clConv2.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clConv2.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

	cl_uint lImageWidth = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	lGlobalSize[0] = ceil((float)lImageWidth/lLocalSize[0])*lLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/lLocalSize[1])*lLocalSize[1];

    cl_int lFilter2DWidth = bfFilter.dim(0);
    if (lFilter2DWidth %2 == 0)
    {
        Log(ERR, this) << "Failure in call to oclFilter2D::conv2 : convolution dimension 0 must be odd ";
    }
    cl_int lFilter2DHeight = bfFilter.dim(1);
    if (lFilter2DWidth %2 == 0)
    {
        Log(ERR, this) << "Failure in call to oclFilter2D::conv2 : convolution dimension 1 must be odd ";
    }

	clSetKernelArg(clConv2, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clConv2, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clConv2, 2, sizeof(cl_int4),  &iAxis);
	clSetKernelArg(clConv2, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clConv2, 4, sizeof(cl_int), &lFilter2DSize);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clConv2, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clConv2.getEvent());
	ENQUEUE_VALIDATE
    */
	return true;
}   



int oclFilter2D::bilateral(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float4 iScalar)
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


int oclFilter2D::sobel(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDx, oclImage2D& bfDy)
{
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clSobel.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clSobel.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

	cl_uint lImageWidth = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	lGlobalSize[0] = ceil((float)lImageWidth/lLocalSize[0])*lLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/lLocalSize[1])*lLocalSize[1];

	clSetKernelArg(clSobel, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clSobel, 1, sizeof(cl_mem), bfDx);
	clSetKernelArg(clSobel, 2, sizeof(cl_mem), bfDy);
 	clSetKernelArg(clSobel, 3, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clSobel, 4, sizeof(cl_uint), &lImageHeight);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clSobel, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clSobel.getEvent());
	ENQUEUE_VALIDATE
	return true;
}


int oclFilter2D::tangent(oclDevice& iDevice, oclImage2D& bfDx, oclImage2D& bfDy, oclImage2D& bfDest)
{
	size_t lLocalSize[2];
	lLocalSize[0] = floor(sqrt(1.0*clTangent.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));
	lLocalSize[1] = floor(sqrt(1.0*clTangent.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice)));

	cl_uint lImageWidth = bfDest.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfDest.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	lGlobalSize[0] = ceil((float)lImageWidth/lLocalSize[0])*lLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/lLocalSize[1])*lLocalSize[1];

	clSetKernelArg(clTangent, 0, sizeof(cl_mem), bfDx);
	clSetKernelArg(clTangent, 1, sizeof(cl_mem), bfDy);
	clSetKernelArg(clTangent, 2, sizeof(cl_mem), bfDest);
 	clSetKernelArg(clTangent, 3, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clTangent, 4, sizeof(cl_uint), &lImageHeight);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clTangent, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clTangent.getEvent());
	ENQUEUE_VALIDATE
	return true;
}
