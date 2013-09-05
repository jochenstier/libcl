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

#include "oclBilinearPyramid.h"

oclBilinearPyramid::oclBilinearPyramid(oclContext& iContext, oclProgram* iParent)
: oclProgram(iContext, "oclBilinearPyramid", iParent)
// kernels
, clUpsample(*this, "clUpsample")
, clDownsample(*this, "clDownsample")
, mLevel(1)
{
    addSourceFile("filter/oclBilinearPyramid.cl");

    cl_image_format lFormat = { CL_RGBA,  CL_HALF_FLOAT };
    mLevel[0] = new oclImage2D(mContext, "Level");
    mLevel[0]->create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, lFormat, 256, 256);
}

//
//
//

int oclBilinearPyramid::compute(oclDevice& iDevice, oclImage2D& bfSrce)
{
    cl_uint lIw = pow(2.0f, ceil(log(bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH)/2.0f)/log(2.0f)));
    cl_uint lIh = pow(2.0f, ceil(log(bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT)/2.0f)/log(2.0f)));
    unsigned int lLevels = log(1.0*min(lIw,lIh))/log(2.0f);

    // resize pyramid if necessary
    cl_uint lLw = mLevel[0]->getImageInfo<size_t>(CL_IMAGE_WIDTH)/2;
    cl_uint lLh = mLevel[0]->getImageInfo<size_t>(CL_IMAGE_HEIGHT)/2;
    if (lLevels != mLevel.size() || lLw != lIw || lLh != lIh )
    {
        for (unsigned int i=0; i<mLevel.size(); i++)
        {
            delete mLevel[i];
        }
        mLevel.resize(lLevels);
        cl_image_format lFormat = { CL_RGBA,  CL_HALF_FLOAT };
        lLw = lIw;
        lLh = lIh;
        for (unsigned int i=0; i<lLevels; i++)
        {
            mLevel[i] = new oclImage2D(mContext, "Level");
            mLevel[i]->create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, lFormat, lLw, lLh);
            lLw /=2;
            lLh /=2;
        }
    }

    // fill in pyramid
    size_t lGlobalSize[2];
    size_t lLocalSize[2];
    clDownsample.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);
    clSetKernelArg(clDownsample, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clDownsample, 1, sizeof(cl_mem), *mLevel[0]);
    clSetKernelArg(clDownsample, 2, sizeof(cl_uint), &lIw);
    clSetKernelArg(clDownsample, 3, sizeof(cl_uint), &lIh);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clDownsample, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clDownsample.getEvent());
    ENQUEUE_VALIDATE

    for (unsigned int i=1; i<mLevel.size(); i++)
    {
        lIw /= 2;
        lIh /= 2;
        clDownsample.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);
        clSetKernelArg(clDownsample, 0, sizeof(cl_mem), *mLevel[i-1]);
        clSetKernelArg(clDownsample, 1, sizeof(cl_mem), *mLevel[i]);
        clSetKernelArg(clDownsample, 2, sizeof(cl_uint), &lIw);
        clSetKernelArg(clDownsample, 3, sizeof(cl_uint), &lIh);
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clDownsample, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clDownsample.getEvent());
        ENQUEUE_VALIDATE

        lIw *= 2;
        lIh *= 2;
        clUpsample.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);
        clSetKernelArg(clUpsample, 0, sizeof(cl_mem), *mLevel[i]);
        clSetKernelArg(clUpsample, 1, sizeof(cl_mem), *mLevel[i-1]);
        clSetKernelArg(clUpsample, 2, sizeof(cl_uint), &lIw);
        clSetKernelArg(clUpsample, 3, sizeof(cl_uint), &lIh);
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clUpsample, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clUpsample.getEvent());
        ENQUEUE_VALIDATE
        lIw /= 2;
        lIh /= 2;
    }

    return true;
}


oclImage2D* oclBilinearPyramid::getLevel(unsigned int iLevel)
{
    // do not return last level. It is useless
    if (iLevel < mLevel.size()-1)
    {
        return mLevel[iLevel];
    }
    else
    {
        return 0;
    }
};
