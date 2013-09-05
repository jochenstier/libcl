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

oclToneMapping::oclToneMapping(oclContext& iContext, oclProgram* iParent)
: oclProgram(iContext, "oclToneMapping", iParent)
// buffers
, bfTempA(iContext, "bfTempA")
, bfTempB(iContext, "bfTempB")
// kernels
, clCombine(*this, "clCombine")
// programs
, mColor(iContext, this)
, mPyramid(iContext, this)
, mMemory(iContext, this)
{
    cl_image_format lFormat = { CL_RGBA,  CL_HALF_FLOAT };
    bfTempA.create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, lFormat, 256, 256);
    bfTempB.create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, lFormat, 256, 256);

    addSourceFile("image/oclToneMapping.cl");
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

    mColor.RGBtoLAB(iDevice, bfSrce, bfTempA);
    mPyramid.compute(iDevice, bfTempA);
    mMemory.memSet(iDevice, bfTempB, mMemory.c0000);

    size_t lGlobalSize[2];
    lGlobalSize[0] = lWidth;
    lGlobalSize[1] = lHeight;
    oclImage2D* bfLevel0 = mPyramid.getLevel(0);
    oclImage2D* bfLevel1 = mPyramid.getLevel(1);
    cl_int lLevel = 1;
    while (bfLevel1 != 0)
    {
        clSetKernelArg(clCombine, 0, sizeof(cl_mem), bfTempA);
        clSetKernelArg(clCombine, 1, sizeof(cl_mem), bfTempB);
        clSetKernelArg(clCombine, 2, sizeof(cl_mem), *bfLevel0);
        clSetKernelArg(clCombine, 3, sizeof(cl_mem), *bfLevel1);
        clSetKernelArg(clCombine, 4, sizeof(cl_int), &lLevel);
        clSetKernelArg(clCombine, 5, sizeof(cl_mem), bfTempB);
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clCombine, 2, NULL, lGlobalSize, NULL, 0, NULL, clCombine.getEvent());
        ENQUEUE_VALIDATE

        lLevel++;
        bfLevel0 = bfLevel1;
        bfLevel1 = mPyramid.getLevel(lLevel);
    }

    mColor.LABtoRGB(iDevice, bfTempB, bfDest);

    return true;
}



/*
    if (bfTempA.getImageInfo<size_t>(CL_IMAGE_WIDTH) != lWidth || bfTempA.getImageInfo<size_t>(CL_IMAGE_HEIGHT) != lHeight)
    {
        bfTempA.resize(lWidth, lHeight);
    }
    if (bfTempB.getImageInfo<size_t>(CL_IMAGE_WIDTH) != lWidth || bfTempB.getImageInfo<size_t>(CL_IMAGE_HEIGHT) != lHeight)
    {
        bfTempB.resize(lWidth, lHeight);
    }
    if (bfTempC.getImageInfo<size_t>(CL_IMAGE_WIDTH) != lWidth || bfTempB.getImageInfo<size_t>(CL_IMAGE_HEIGHT) != lHeight)
    {
        bfTempC.resize(lWidth, lHeight);
    }
*/