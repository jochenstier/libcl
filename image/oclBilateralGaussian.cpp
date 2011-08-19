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
// limitations under the License.#ifndef _oclBilateralGaussian
#include <math.h>

#include "oclBilateralGaussian.h"


oclBilateralGaussian::oclBilateralGaussian(oclContext& iContext)
: oclProgram(iContext, "oclBilateralGaussian")
, clBilateralGaussian(*this)
{
	addSourceFile("image\\oclBilateralGaussian.cl");

	exportKernel(clBilateralGaussian);
}

//
//
//

int oclBilateralGaussian::compile()
{
	clBilateralGaussian = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clBilateralGaussian = createKernel("clBilateralGaussian");
	KERNEL_VALIDATE(clBilateralGaussian)
	mLocalSize[0] = floor(sqrt(1.0*clBilateralGaussian.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, mContext.getDevice(0))));
	mLocalSize[1] = floor(sqrt(1.0*clBilateralGaussian.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, mContext.getDevice(0))));
	setRadius(2);
	setScalar(0.2f);
	return 1;
}

//
//
//

void oclBilateralGaussian::setRadius(cl_uint iValue)
{
	clSetKernelArg(clBilateralGaussian, 2, sizeof(cl_uint), &iValue);
}

void oclBilateralGaussian::setScalar(cl_float iValue)
{
	clSetKernelArg(clBilateralGaussian, 3, sizeof(cl_float), &iValue);
}

int oclBilateralGaussian::compute(oclDevice& iDevice, oclImage2D& bfSource, oclImage2D& bfDest)
{
	size_t lGlobalSize[2];
	cl_uint lImageWidth;
	cl_uint lImageHeight;

	lImageWidth = bfSource.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	lImageHeight = bfSource.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	lGlobalSize[0] = ceil((float)lImageWidth/mLocalSize[0])*mLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/mLocalSize[1])*mLocalSize[1];
 	clSetKernelArg(clBilateralGaussian, 4, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clBilateralGaussian, 5, sizeof(cl_uint), &lImageHeight);
	clSetKernelArg(clBilateralGaussian, 0, sizeof(cl_mem), bfSource);
	clSetKernelArg(clBilateralGaussian, 1, sizeof(cl_mem), bfDest);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clBilateralGaussian, 2, NULL, lGlobalSize, mLocalSize, 0, NULL, clBilateralGaussian.getEvent());
	ENQUEUE_VALIDATE

	lImageWidth = bfDest.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	lImageHeight = bfDest.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	lGlobalSize[0] = ceil((float)lImageWidth/mLocalSize[0])*mLocalSize[0];
	lGlobalSize[1] = ceil((float)lImageHeight/mLocalSize[1])*mLocalSize[1];
 	clSetKernelArg(clBilateralGaussian, 4, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clBilateralGaussian, 5, sizeof(cl_uint), &lImageHeight);
	clSetKernelArg(clBilateralGaussian, 0, sizeof(cl_mem), bfDest);
	clSetKernelArg(clBilateralGaussian, 1, sizeof(cl_mem), bfSource);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clBilateralGaussian, 2, NULL, lGlobalSize, mLocalSize, 0, NULL, clBilateralGaussian.getEvent());
	ENQUEUE_VALIDATE

	return true;
}

