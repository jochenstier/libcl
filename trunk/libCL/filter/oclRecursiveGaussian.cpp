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
// limitations under the License.#ifndef _oclRecursiveGaussian
#include <math.h>

#include "oclRecursiveGaussian.h"

oclRecursiveGaussian::oclRecursiveGaussian(oclContext& iContext)
: oclProgram(iContext, "oclRecursiveGaussian")
// kernels
, clRecursiveGaussian(*this)
{
	addSourceFile("filter\\oclRecursiveGaussian.cl");

	exportKernel(clRecursiveGaussian);
}

//
//
//

int oclRecursiveGaussian::compile()
{
	clRecursiveGaussian = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clRecursiveGaussian = createKernel("clRecursiveGaussian");
	KERNEL_VALIDATE(clRecursiveGaussian)
	mLocalSize = clRecursiveGaussian.getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, mContext.getDevice(0));
	setSigma(10.1f);

	return 1;
}

//
//
//

void oclRecursiveGaussian::setSigma(cl_float nsigma)
{
    cl_float alpha = 1.695f / nsigma;
    cl_float ema = exp(-alpha);
    cl_float ema2 = exp(-2.0f * alpha);
    cl_float b1 = -2.0f * ema;
    cl_float b2 = ema2;

	cl_float k = (1.0f - ema)*(1.0f - ema)/(1.0f + (2.0f * alpha * ema) - ema2);
	cl_float a0 = k;
	cl_float a1 = k * (alpha - 1.0f) * ema;
	cl_float a2 = k * (alpha + 1.0f) * ema;
	cl_float a3 = -k * ema2;
    cl_float coefp = (a0 + a1)/(1.0f + b1 + b2);
    cl_float coefn = (a2 + a3)/(1.0f + b1 + b2);

	clSetKernelArg(clRecursiveGaussian, 5, sizeof(cl_float), &a0);
	clSetKernelArg(clRecursiveGaussian, 6, sizeof(cl_float), &a1);
	clSetKernelArg(clRecursiveGaussian, 7, sizeof(cl_float), &a2);
	clSetKernelArg(clRecursiveGaussian, 8, sizeof(cl_float), &a3);
	clSetKernelArg(clRecursiveGaussian, 9, sizeof(cl_float), &b1);
	clSetKernelArg(clRecursiveGaussian, 10, sizeof(cl_float), &b2);
	clSetKernelArg(clRecursiveGaussian, 11, sizeof(cl_float), &coefp);
	clSetKernelArg(clRecursiveGaussian, 12, sizeof(cl_float), &coefn);
}

int oclRecursiveGaussian::compute(oclDevice& iDevice, oclImage2D& bfSource, oclImage2D& bfTemp, oclImage2D& bfDest)
{
	size_t lGlobalSize;
	cl_uint2 dxy;
	cl_uint lImageWidth = bfSource.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lImageHeight = bfSource.getImageInfo<size_t>(CL_IMAGE_HEIGHT);

 	clSetKernelArg(clRecursiveGaussian, 2, sizeof(cl_uint), &lImageWidth);
	clSetKernelArg(clRecursiveGaussian, 3, sizeof(cl_uint), &lImageHeight);

	dxy.s[0] = 0;
	dxy.s[1] = 1;
	lGlobalSize = ceil((float)lImageWidth/mLocalSize)*mLocalSize;
	clSetKernelArg(clRecursiveGaussian, 0, sizeof(cl_mem), bfSource);
	clSetKernelArg(clRecursiveGaussian, 1, sizeof(cl_mem), bfTemp);
	clSetKernelArg(clRecursiveGaussian, 4, sizeof(cl_uint2), &dxy);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clRecursiveGaussian, 1, NULL, &lGlobalSize, &mLocalSize, 0, NULL, clRecursiveGaussian.getEvent());
	ENQUEUE_VALIDATE

	dxy.s[0] = 1;
	dxy.s[1] = 0;
	lGlobalSize = ceil((float)lImageHeight/mLocalSize)*mLocalSize;
	clSetKernelArg(clRecursiveGaussian, 0, sizeof(cl_mem), bfTemp);
	clSetKernelArg(clRecursiveGaussian, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clRecursiveGaussian, 4, sizeof(cl_uint2), &dxy);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clRecursiveGaussian, 1, NULL, &lGlobalSize, &mLocalSize, 0, NULL, clRecursiveGaussian.getEvent());
	ENQUEUE_VALIDATE

	return true;
}