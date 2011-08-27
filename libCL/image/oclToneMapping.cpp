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
#include "oclToneMapping.h"

oclToneMapping::oclToneMapping(oclContext& iContext, cl_image_format iFormat)
: oclProgram(iContext, "oclToneMapping")
// buffers
, bfTempA(iContext, "bfTemp0")
, bfTempB(iContext, "bfTemp1")
// kernels
, clLuminance(*this)
, clCombine(*this)
// programs
, mGaussian(iContext)
{
    bfTempA.create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iFormat, 256, 256);
    bfTempB.create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iFormat, 256, 256);

    addSourceFile("image\\oclToneMapping.cl");

    exportKernel(clLuminance);
    exportKernel(clCombine);
}

int oclToneMapping::compile()
{
    // release kernels
    clLuminance = 0;
    clCombine = 0;

    if (!mGaussian.compile())
    {
        return 0;
    }
    if (!oclProgram::compile())
    {
        return 0;
    }

    clLuminance = createKernel("clLuminance");
    KERNEL_VALIDATE(clLuminance)
    clCombine = createKernel("clCombine");
    KERNEL_VALIDATE(clCombine)
    return 1;
}

int oclToneMapping::compute(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
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

    size_t lGlobalSize[2];
    lGlobalSize[0] = lWidth;
    lGlobalSize[1] = lHeight;

    clSetKernelArg(clLuminance, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clLuminance, 1, sizeof(cl_mem), bfTempA);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clLuminance, 2, NULL, lGlobalSize, NULL, 0, NULL, clLuminance.getEvent());
    ENQUEUE_VALIDATE

    if (!mGaussian.compute(iDevice, bfTempA, bfTempB, bfTempA))
    {
        return false;
    }

    clSetKernelArg(clCombine, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clCombine, 1, sizeof(cl_mem), bfTempA);
    clSetKernelArg(clCombine, 2, sizeof(cl_mem), bfDest);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clCombine, 2, NULL, lGlobalSize, NULL, 0, NULL, clCombine.getEvent());
    ENQUEUE_VALIDATE

    return true;
}


void oclToneMapping::setSmoothing(cl_float iValue)
{
    mGaussian.setSigma(iValue);
}

cl_image_format oclToneMapping::sDefaultFormat = { CL_RGBA,  CL_HALF_FLOAT };
