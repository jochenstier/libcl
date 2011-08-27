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
#include "oclImage2D.h"

oclImage2D::oclImage2D(oclContext& iContext, char* iName)
: oclMem(iContext, iName)
{
};

bool oclImage2D::create(cl_mem_flags iMemFlags, cl_image_format& iFormat, size_t iDim0, size_t iDim1, void* iHostPtr)
{
    mMemPtr = clCreateImage2D(mContext, iMemFlags, &iFormat, iDim0, iDim1, 0, iHostPtr, &sStatusCL);
    return oclSuccess("clCreateImage2D", this);
}

bool oclImage2D::resize(size_t iDim0, size_t iDim1, void* iHostPtr) 
{
    if (mMemPtr)
    {
        cl_mem_flags lMemFlags = getMemObjectInfo<cl_mem_flags>(CL_MEM_FLAGS); 
        cl_image_format lFormat = getImageInfo<cl_image_format>(CL_IMAGE_FORMAT);
        destroy();
        return create(lMemFlags, lFormat, iDim0, iDim1, iHostPtr);
    }
    return false;
}

bool oclImage2D::map(oclDevice& iDevice, cl_map_flags iMapping)
{
    if (mMemPtr)
    {
        size_t lOrigin[3];  
        size_t lRegion[3];

        lOrigin[0] = 0;
        lOrigin[1] = 0;
        lOrigin[2] = 0;

        lRegion[0] = getImageInfo<size_t>(CL_IMAGE_WIDTH);
        lRegion[1] = getImageInfo<size_t>(CL_IMAGE_HEIGHT);
        lRegion[2] = 1;

        size_t lRowpitch;
        size_t lSlicepitch;

        mHostPtr = (unsigned char*)clEnqueueMapImage(iDevice, 
                                                      mMemPtr, 
                                                      CL_TRUE, 
                                                      iMapping, 
                                                      lOrigin, 
                                                      lRegion, 
                                                      &lRowpitch,
                                                      &lSlicepitch,
                                                      0, 
                                                      NULL, 
                                                      NULL, 
                                                      &sStatusCL);
        mMapping = iMapping;
        return oclSuccess("clEnqueueMapBuffer", this);
    }
    Log(ERR, this) << "Invalid cl_mem";
    return false;
}

bool oclImage2D::write(oclDevice& iDevice)
{
    if (mMemPtr)
    {
        size_t lOrigin[3];  
        size_t lRegion[3];

        lOrigin[0] = 0;
        lOrigin[1] = 0;
        lOrigin[2] = 0;

        lRegion[0] = getImageInfo<size_t>(CL_IMAGE_WIDTH);
        lRegion[1] = getImageInfo<size_t>(CL_IMAGE_HEIGHT);
        lRegion[2] = 1;

        size_t lRowpitch = 0; 
        size_t lSlicepitch = 0; 

        sStatusCL = clEnqueueWriteImage (iDevice,
                                         mMemPtr,
                                         CL_TRUE,
                                         lOrigin,
                                         lRegion,
                                         lRowpitch,
                                         lSlicepitch,
                                         mHostPtr,
                                         0,
                                         0,
                                         0);
        return oclSuccess("clEnqueueWriteImage", this);
    }
    Log(ERR, this) << "Invalid cl_mem";
    return false;
}

bool oclImage2D::read(oclDevice& iDevice)
{
    if (mMemPtr)
    {
        size_t lOrigin[3];  
        size_t lRegion[3];

        lOrigin[0] = 0;
        lOrigin[1] = 0;
        lOrigin[2] = 0;

        lRegion[0] = getImageInfo<size_t>(CL_IMAGE_WIDTH);
        lRegion[1] = getImageInfo<size_t>(CL_IMAGE_HEIGHT);
        lRegion[2] = 1;

        size_t lRowpitch = 0; 
        size_t lSlicepitch = 0; 

        sStatusCL = clEnqueueReadImage (iDevice,
                                         mMemPtr,
                                         CL_TRUE,
                                         lOrigin,
                                         lRegion,
                                         lRowpitch,
                                         lSlicepitch,
                                         mHostPtr,
                                         0,
                                         0,
                                         0);
        return oclSuccess("clEnqueueReadImage", this);
    }
    Log(ERR, this) << "Invalid cl_mem";
    return false;
}


size_t oclImage2D::dim(int iAxis)
{
    switch (iAxis)
    {
        case 0: 
            return getImageInfo<size_t>(CL_IMAGE_WIDTH);
        case 1: 
            return getImageInfo<size_t>(CL_IMAGE_HEIGHT);
    }
    return 0;
};
