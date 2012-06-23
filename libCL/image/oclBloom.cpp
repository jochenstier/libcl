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
#include "oclBloom.h"

#include <math.h>

oclBloom::oclBloom(oclContext& iContext, cl_image_format iFormat)
: oclProgram(iContext, "oclBloom")
, mGaussian(iContext)
// buffers
, bfTempA(iContext, "bfTemp0")
, bfTempB(iContext, "bfTemp1")
// kernels
, clFilter(*this)
, clCombine(*this)
{
    bfTempA.create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iFormat, 256, 256);
    bfTempB.create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iFormat, 256, 256);

    addSourceFile("image/oclBloom.cl");

    exportKernel(clFilter);
    exportKernel(clCombine);
}

int oclBloom::compile()
{
    clFilter = 0;
    clCombine = 0;

    if (!mGaussian.compile())
    {
        return 0;
    }
    if (!oclProgram::compile())
    {
        return 0;
    }

    clFilter = createKernel("clFilter");
    KERNEL_VALIDATE(clFilter)
    setThreshold(0.9f);

    clCombine = createKernel("clCombine");
    KERNEL_VALIDATE(clCombine)
    setIntensity(0.9f);

    return 1;
}


int oclBloom::compute(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
{
    cl_uint lWidth = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
    cl_uint lHeight = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);

    if (bfTempA.getImageInfo<size_t>(CL_IMAGE_WIDTH) != lWidth || bfTempA.getImageInfo<size_t>(CL_IMAGE_HEIGHT) != lHeight)
    {
        bfTempA.resize(lWidth, lHeight);
    }
    if (bfTempB.getImageInfo<size_t>(CL_IMAGE_WIDTH) != lWidth || bfTempB.getImageInfo<size_t>(CL_IMAGE_HEIGHT) != lHeight)
    {
        bfTempB.resize(lWidth, lHeight);
    }

    size_t lGlobalWorkSize[2];
    lGlobalWorkSize[0] = lWidth;
    lGlobalWorkSize[1] = lHeight;

    clSetKernelArg(clFilter, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clFilter, 1, sizeof(cl_mem), bfTempA);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clFilter, 2, NULL, lGlobalWorkSize, NULL, 0, NULL, clFilter.getEvent());
    ENQUEUE_VALIDATE

    if (!mGaussian.compute(iDevice, bfTempA, bfTempB, bfTempA))
    {
        return false;
    }

    clSetKernelArg(clCombine, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clCombine, 1, sizeof(cl_mem), bfTempA);
    clSetKernelArg(clCombine, 2, sizeof(cl_mem), bfDest);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clCombine, 2, NULL, lGlobalWorkSize, NULL, 0, NULL, clCombine.getEvent());
    ENQUEUE_VALIDATE

    return true;
}

void oclBloom::setSmoothing(cl_float iValue)
{
    mGaussian.setSigma(iValue);
}

void oclBloom::setThreshold(cl_float iValue)
{
    clSetKernelArg(clFilter, 2, sizeof(cl_float), &iValue);
}

void oclBloom::setIntensity(cl_float iValue)
{
    clSetKernelArg(clCombine, 3, sizeof(cl_float), &iValue);
}

cl_image_format oclBloom::sDefaultFormat = { CL_RGBA,  CL_HALF_FLOAT };
