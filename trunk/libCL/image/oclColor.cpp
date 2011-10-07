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
// limitations under the License.#ifndef _oclColor
#include <math.h>

#include "oclColor.h"

oclColor::oclColor(oclContext& iContext)
: oclProgram(iContext, "oclColor")
// kernels
, clHSVtoRGB(*this)
, clRGBtoHSV(*this)
{
    addSourceFile("image\\oclColor.cl");
}

//
//
//

int oclColor::compile()
{
	clHSVtoRGB = 0;
	clRGBtoHSV = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clHSVtoRGB = createKernel("clHSVtoRGB");
	KERNEL_VALIDATE(clHSVtoRGB)
	clRGBtoHSV = createKernel("clRGBtoHSV");
	KERNEL_VALIDATE(clRGBtoHSV)
	return 1;
}

//
//
//
int oclColor::HSVtoRGB(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfSrce.dim(0);
    lGlobalSize[1] = bfSrce.dim(1);
	clSetKernelArg(clHSVtoRGB, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clHSVtoRGB, 1, sizeof(cl_mem), bfDest);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clHSVtoRGB, 2, NULL, lGlobalSize, 0, 0, NULL, clHSVtoRGB.getEvent());
	ENQUEUE_VALIDATE
    return 1;
};

int oclColor::RGBtoHSV(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfSrce.dim(0);
    lGlobalSize[1] = bfSrce.dim(1);
	clSetKernelArg(clRGBtoHSV, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clRGBtoHSV, 1, sizeof(cl_mem), bfDest);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clRGBtoHSV, 3, NULL, lGlobalSize, 0, 0, NULL, clRGBtoHSV.getEvent());
	ENQUEUE_VALIDATE
    return 1;
};