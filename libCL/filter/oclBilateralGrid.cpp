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

#include "oclBilateralGrid.h"

oclBilateralGrid::oclBilateralGrid(oclContext& iContext)
: oclProgram(iContext, "oclBilateralGrid")
// kernels
, bfGrid1Da(iContext, "bfGrid1Da")
, bfGrid1Db(iContext, "bfGrid1Db")
, bfGrid3D(iContext, "bfTempB")
// kernels
, clSplit(*this)
, clSlice(*this)
, clEqualize(*this)
, clConvolute(*this)
// programs
, mMemory(iContext)
// vars
, bfCurr(&bfGrid1Da)
, bfTemp(&bfGrid1Db)
{
    mGridSize[0] = 16;
    mGridSize[1] = 16;
    mGridSize[2] = 32;

    cl_image_format lFormat = { CL_RGBA,  CL_FLOAT };
    bfGrid3D.create(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, lFormat, mGridSize[0], mGridSize[1], mGridSize[2]);
    bfGrid1Da.create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mGridSize[0]*mGridSize[1]*mGridSize[2]);
    bfGrid1Db.create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mGridSize[0]*mGridSize[1]*mGridSize[2]);

    
    addSourceFile("filter/oclBilateralGrid.cl");

    exportKernel(clSplit);
    exportKernel(clSlice);
    exportKernel(clEqualize);
}

//
//
//

int oclBilateralGrid::compile()
{
    clSplit = 0;
    clSlice = 0;
    clEqualize = 0;

    if (!oclProgram::compile() || !mMemory.compile())
    {
        return 0;
    }

    clSplit = createKernel("clSplit");
    KERNEL_VALIDATE(clSplit)
    clSlice = createKernel("clSlice");
    KERNEL_VALIDATE(clSlice)
    clEqualize = createKernel("clEqualize");
    KERNEL_VALIDATE(clEqualize)
    clConvolute = createKernel("clConvolute");
    KERNEL_VALIDATE(clConvolute)
    return 1;
}

//
//
//

void oclBilateralGrid::resize(cl_uint iGridW, cl_uint iGridH, cl_uint iGridD)
{
    mGridSize[0] = iGridW;
    mGridSize[1] = iGridH;
    mGridSize[2] = iGridD;
    bfGrid3D.resize(mGridSize[0], mGridSize[1], mGridSize[2]);
    bfGrid1Da.resize<cl_float4>(mGridSize[0]*mGridSize[1]*mGridSize[2]);
    bfGrid1Db.resize<cl_float4>(mGridSize[0]*mGridSize[1]*mGridSize[2]);
}

//
//
//

int oclBilateralGrid::split(oclDevice& iDevice, oclImage2D& bfSrce, cl_float4 iMask)
{
    cl_uint lImageW = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
    cl_uint lImageH = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
    if (lImageW%mGridSize[0] != 0 || lImageH%mGridSize[1] != 0)
    {
        Log(WARN, this) << "Image dimensions should be divisible by grid dimensions";
    }
    lImageH /= mGridSize[1];
    lImageW /= mGridSize[0];

    mMemory.memSet(iDevice, bfGrid1Da, oclMemory::c0000);

    bfCurr = &bfGrid1Da;
    clSetKernelArg(clSplit, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clSplit, 1, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clSplit, 2, sizeof(cl_uint), &lImageW);
    clSetKernelArg(clSplit, 3, sizeof(cl_uint), &lImageH);
    clSetKernelArg(clSplit, 4, sizeof(cl_float4), &iMask);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clSplit, 3, NULL, mGridSize, NULL, 0, NULL, clSplit.getEvent());
    ENQUEUE_VALIDATE

    return true;
}

int oclBilateralGrid::slice(oclDevice& iDevice, oclImage2D& bfSrce, cl_float4 iMask, oclImage2D& bfDest)
{
    size_t origin[3] = { 0, 0, 0,}; 
    size_t offset = 0; 
    size_t region[3] = { mGridSize[0], mGridSize[1], mGridSize[2] }; 

    sStatusCL = clEnqueueCopyBufferToImage (iDevice,
                                            *bfCurr,
                                            bfGrid3D,
                                            offset,
                                            origin,
                                            region,
                                            0,
                                            0,
                                            0);
    oclSuccess("clEnqueueCopyBufferToImage", this);

    size_t lGlobalSize[2];
    lGlobalSize[0] = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
    lGlobalSize[1] = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
    clSetKernelArg(clSlice, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clSlice, 1, sizeof(cl_float4), &iMask);
    clSetKernelArg(clSlice, 2, sizeof(cl_mem), bfDest);
    clSetKernelArg(clSlice, 3, sizeof(cl_mem), bfGrid3D);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clSlice, 2, NULL, lGlobalSize, NULL, 0, NULL, clSlice.getEvent());
    ENQUEUE_VALIDATE

    return true;
}


int oclBilateralGrid::equalize(oclDevice& iDevice, cl_float4 iMask)
{
    size_t lLocalSize[2];
    lLocalSize[0] = 1;
    lLocalSize[1] = 1;
    clSetKernelArg(clEqualize, 0, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clEqualize, 1, sizeof(cl_float4), &iMask);
    clSetKernelArg(clEqualize, 2, sizeof(cl_int), &mGridSize[2]);
    clSetKernelArg(clEqualize, 3, sizeof(cl_float)*mGridSize[2], 0);
    clSetKernelArg(clEqualize, 4, sizeof(cl_float4)*mGridSize[2], 0);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clEqualize, 2, NULL, mGridSize, lLocalSize, 0, NULL, clEqualize.getEvent());
    ENQUEUE_VALIDATE

    return 1;
}


cl_int4 oclBilateralGrid::sAxisX = { 1, 0, 0, 0 };
cl_int4 oclBilateralGrid::sAxisY = { 0, 1, 0, 0 };
cl_int4 oclBilateralGrid::sAxisZ = { 0, 0, 1, 0 };

int oclBilateralGrid::smoothXY(oclDevice& iDevice, oclBuffer& iFilter)
{
    cl_int lFilterSize = iFilter.dim(0)/sizeof(cl_float);
    clSetKernelArg(clConvolute, 3, sizeof(cl_mem), iFilter);
    clSetKernelArg(clConvolute, 4, sizeof(cl_int), &lFilterSize);

    clSetKernelArg(clConvolute, 0, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clConvolute, 1, sizeof(cl_mem), *bfTemp);
    clSetKernelArg(clConvolute, 2, sizeof(cl_int4),  &sAxisX);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clConvolute, 3, NULL, mGridSize, 0, 0, NULL, clConvolute.getEvent());
    ENQUEUE_VALIDATE

    clSetKernelArg(clConvolute, 0, sizeof(cl_mem), *bfTemp);
    clSetKernelArg(clConvolute, 1, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clConvolute, 2, sizeof(cl_int4),  &sAxisY);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clConvolute, 3, NULL, mGridSize, 0, 0, NULL, clConvolute.getEvent());
    ENQUEUE_VALIDATE

    //mConvolute.iso3Dsep(iDevice, *bfCurr, *bfTemp, mGridSize, sAxisX, iFilter);
    //mConvolute.iso3Dsep(iDevice, *bfTemp, *bfCurr, mGridSize, sAxisY, iFilter);
    return 1;
}

int oclBilateralGrid::smoothXYZ(oclDevice& iDevice, oclBuffer& iFilter)
{
    cl_int lFilterSize = iFilter.dim(0)/sizeof(cl_float);
    clSetKernelArg(clConvolute, 3, sizeof(cl_mem), iFilter);
    clSetKernelArg(clConvolute, 4, sizeof(cl_int), &lFilterSize);

    clSetKernelArg(clConvolute, 0, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clConvolute, 1, sizeof(cl_mem), *bfTemp);
    clSetKernelArg(clConvolute, 2, sizeof(cl_int4),  &sAxisX);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clConvolute, 3, NULL, mGridSize, 0, 0, NULL, clConvolute.getEvent());
    ENQUEUE_VALIDATE

    clSetKernelArg(clConvolute, 0, sizeof(cl_mem), *bfTemp);
    clSetKernelArg(clConvolute, 1, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clConvolute, 2, sizeof(cl_int4),  &sAxisY);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clConvolute, 3, NULL, mGridSize, 0, 0, NULL, clConvolute.getEvent());
    ENQUEUE_VALIDATE

    clSetKernelArg(clConvolute, 0, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clConvolute, 1, sizeof(cl_mem), *bfTemp);
    clSetKernelArg(clConvolute, 2, sizeof(cl_int4),  &sAxisZ);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clConvolute, 3, NULL, mGridSize, 0, 0, NULL, clConvolute.getEvent());
    ENQUEUE_VALIDATE

    //mConvolute.iso3Dsep(iDevice, *bfCurr, *bfTemp, mGridSize, sAxisX, iFilter);
    //mConvolute.iso3Dsep(iDevice, *bfTemp, *bfCurr, mGridSize, sAxisY, iFilter);
    //mConvolute.iso3Dsep(iDevice, *bfCurr, *bfTemp, mGridSize, sAxisZ, iFilter);

    oclBuffer* lTemp = bfCurr;
    bfCurr = bfTemp;
    bfTemp = lTemp;
    return 1;
};

int oclBilateralGrid::smoothZ(oclDevice& iDevice, oclBuffer& iFilter)
{
    cl_int lFilterSize = iFilter.dim(0)/sizeof(cl_float);
    clSetKernelArg(clConvolute, 3, sizeof(cl_mem), iFilter);
    clSetKernelArg(clConvolute, 4, sizeof(cl_int), &lFilterSize);

    clSetKernelArg(clConvolute, 0, sizeof(cl_mem), *bfCurr);
    clSetKernelArg(clConvolute, 1, sizeof(cl_mem), *bfTemp);
    clSetKernelArg(clConvolute, 2, sizeof(cl_int4),  &sAxisX);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clConvolute, 3, NULL, mGridSize, 0, 0, NULL, clConvolute.getEvent());
    ENQUEUE_VALIDATE

    //mConvolute.iso3Dsep(iDevice, *bfCurr, *bfTemp, mGridSize, sAxisZ, iFilter);

    oclBuffer* lTemp = bfCurr;
    bfCurr = bfTemp;
    bfTemp = lTemp;
    return 1;
};
