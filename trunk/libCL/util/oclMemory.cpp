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

#include "oclMemory.h"


cl_float4 oclMemory::c0000 = { 0, 0, 0, 0 };


oclMemory::oclMemory(oclContext& iContext, oclProgram* iParent)
: oclProgram(iContext, "oclMemory", iParent)
// kernels
, clMemSetImage(*this, "clMemSetImage")
, clMemSetBuffer_float4(*this, "clMemSetBuffer_float4")
, clMemSetBuffer_float(*this, "clMemSetBuffer_float")
, clMean(*this, "clMean")
, clMin(*this, "clMin")
, clMax(*this, "clMax")
, clVariance(*this, "clVariance")
, clNormalize(*this, "clNormalize")
//, clTemp(*this, "clTemp")

, bfMean(iContext, "bfMean", oclBuffer::_float)
, bfVariance(iContext, "bfVariance", oclBuffer::_float)
{
	bfMean.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
	bfVariance.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);

    addSourceFile("util/oclMemory.cl");

	oclDevice& lDevice = iContext.getDevice(0);
	clGetDeviceInfo(lDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(mMaxWorkgroupSizes), &mMaxWorkgroupSizes, NULL);
}


int oclMemory::memSet(oclDevice& iDevice, oclImage2D& bfDest, cl_float4 iValue)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfDest.dim(0);
    lGlobalSize[1] = bfDest.dim(1);
    clSetKernelArg(clMemSetImage, 0, sizeof(cl_mem), bfDest);
    clSetKernelArg(clMemSetImage, 1, sizeof(cl_float4), &iValue);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clMemSetImage, 2, NULL, lGlobalSize, 0, 0, NULL, clMemSetImage.getEvent());
    ENQUEUE_VALIDATE
    return 1;
};


int oclMemory::memSet(oclDevice& iDevice, oclBuffer& bfDest, cl_float4 iValue)
{
    size_t lGlobalSize;
    lGlobalSize = bfDest.dim(0)/sizeof(cl_float4);
    clSetKernelArg(clMemSetBuffer_float4, 0, sizeof(cl_mem), bfDest);
    clSetKernelArg(clMemSetBuffer_float4, 1, sizeof(cl_float4), &iValue);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clMemSetBuffer_float4, 1, NULL, &lGlobalSize, 0, 0, NULL, clMemSetBuffer_float4.getEvent());
    ENQUEUE_VALIDATE
    return 1;
};

int oclMemory::memSet(oclDevice& iDevice, oclBuffer& bfDest, cl_float iValue)
{
    size_t lGlobalSize;
    lGlobalSize = bfDest.dim(0)/sizeof(cl_float);
    clSetKernelArg(clMemSetBuffer_float, 0, sizeof(cl_mem), bfDest);
    clSetKernelArg(clMemSetBuffer_float, 1, sizeof(cl_float), &iValue);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clMemSetBuffer_float, 1, NULL, &lGlobalSize, 0, 0, NULL, clMemSetBuffer_float.getEvent());
    ENQUEUE_VALIDATE
    return 1;
}; 


 
float debugsum(oclBuffer& iBuffer)
{
	float lSum = 0;
	if (iBuffer.map(CL_MAP_READ))
	{
		int lDim = iBuffer.count<cl_float>();
		cl_float* lPtr = iBuffer.ptr<cl_float>();
		for (int i=0; i<lDim; i++)
		{
			lSum += lPtr[i];
		}
		iBuffer.unmap(); 
	}
	return lSum;
}
int oclMemory::mean(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest)
{
    cl_uint lSize = bfSrce.count<cl_float>();
    cl_uint lCount = lSize/mMaxWorkgroupSizes[0]+1;
    clSetKernelArg(clMean, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clMean, 1, sizeof(cl_mem), bfDest);
    clSetKernelArg(clMean, 2, sizeof(cl_int), &lSize);
    clSetKernelArg(clMean, 3, sizeof(cl_int), &lCount);
    clSetKernelArg(clMean, 4, mMaxWorkgroupSizes[0]*sizeof(cl_float), 0);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clMean, 1, NULL, &mMaxWorkgroupSizes[0], &mMaxWorkgroupSizes[0], 0, NULL, clMean.getEvent());
    ENQUEUE_VALIDATE

// JSTIER find out why there is such a large error between the above and below solution

//Log(INFO) << debugsum(bfDest)/sizeof(cl_float);
/*
Log(INFO) << debugsum(bfDest);

    size_t lGlobalSize = 1;
    lSize = bfSrce.dim(0)/sizeof(cl_float);
    clSetKernelArg(clMean, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clMean, 1, sizeof(cl_mem), bfDest);
    clSetKernelArg(clMean, 2, sizeof(cl_int), &lSize);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clMean, 1, NULL, &lGlobalSize, 0, 0, NULL, clMean.getEvent());
    ENQUEUE_VALIDATE

Log(INFO) << debugsum(bfDest);
*/
    return 1;
};

int oclMemory::variance(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfMean, oclBuffer& bfDest)
{
    cl_uint lSize = bfSrce.count<cl_float>();
    cl_uint lCount = lSize/mMaxWorkgroupSizes[0]+1;
    clSetKernelArg(clVariance, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clVariance, 1, sizeof(cl_mem), bfMean);
    clSetKernelArg(clVariance, 2, sizeof(cl_mem), bfDest);
    clSetKernelArg(clVariance, 3, sizeof(cl_int), &lSize);
    clSetKernelArg(clVariance, 4, sizeof(cl_int), &lCount);
    clSetKernelArg(clVariance, 5, mMaxWorkgroupSizes[0]*sizeof(cl_float), 0);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clVariance, 1, NULL, &mMaxWorkgroupSizes[0], &mMaxWorkgroupSizes[0], 0, NULL, clVariance.getEvent());
    ENQUEUE_VALIDATE

// JSTIER find out why there is such a large error between the above and below solution
/*
    Log(INFO) << "v0  " << debugsum(bfDest);

    size_t lGlobalSize = 1;
    lSize = bfSrce.dim(0)/sizeof(cl_float);
    clSetKernelArg(clVariance, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clVariance, 1, sizeof(cl_mem), bfMean);
    clSetKernelArg(clVariance, 2, sizeof(cl_mem), bfDest);
    clSetKernelArg(clVariance, 3, sizeof(cl_int), &lSize);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clVariance, 1, NULL, &lGlobalSize, 0, 0, NULL, clVariance.getEvent());
    ENQUEUE_VALIDATE

    Log(INFO) << "v0  " << debugsum(bfDest);
*/
    return 1;
};

int oclMemory::normalize(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest)
{
	mean(iDevice, bfSrce, bfMean);
	variance(iDevice, bfSrce, bfMean, bfVariance);

    size_t lGlobalSize = bfSrce.count<cl_float>();
	clSetKernelArg(clNormalize, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clNormalize, 1, sizeof(cl_mem), bfMean);
	clSetKernelArg(clNormalize, 2, sizeof(cl_mem), bfVariance);
	clSetKernelArg(clNormalize, 3, sizeof(cl_mem), bfDest);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clNormalize, 1, NULL, &lGlobalSize, NULL, 0, NULL, clNormalize.getEvent());
	VALIDATE_KENEL(clNormalize)
    return 1;
};


int oclMemory::min(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest)
{
    size_t lGlobalSize = 1;
    cl_uint lSize = bfSrce.dim(0)/sizeof(cl_float);
    clSetKernelArg(clMin, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clMin, 1, sizeof(cl_mem), bfDest);
    clSetKernelArg(clMin, 2, sizeof(cl_int), &lSize);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clMin, 1, NULL, &lGlobalSize, 0, 0, NULL, clMin.getEvent());
    ENQUEUE_VALIDATE
    return 1;
};


int oclMemory::max(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest)
{
    size_t lGlobalSize = 1;
    cl_uint lSize = bfSrce.dim(0)/sizeof(cl_float);
    clSetKernelArg(clMax, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clMax, 1, sizeof(cl_mem), bfDest);
    clSetKernelArg(clMax, 2, sizeof(cl_int), &lSize);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clMax, 1, NULL, &lGlobalSize, 0, 0, NULL, clMax.getEvent());
    ENQUEUE_VALIDATE
    return 1;
};
int oclMemory::random(oclDevice& iDevice, oclBuffer& bfDest, cl_float iMin, cl_float iMax)
{
	if (bfDest.map(CL_MAP_WRITE))
	{
		int lDim = bfDest.count<cl_float>();
		cl_float* lPtr = bfDest.ptr<cl_float>();
		for (int i=0; i<lDim; i++)
		{
			double lRand = (double)rand()/(double)RAND_MAX;
			lPtr[i] = iMin+lRand*(iMax-iMin);
		}
		bfDest.unmap();
	}

    return 1;
};


int oclMemory::sum(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest)
{
    return 1;
};
