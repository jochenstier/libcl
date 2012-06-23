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
#include "oclRadixSort.h"

#include <math.h>

#define CBITS 4
#define BLOCK_SIZE 256
#define BLOCK_SIZE_CUBE BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE

const int oclRadixSort::cBits = CBITS;
const size_t oclRadixSort::cBlockSize = BLOCK_SIZE;
const size_t oclRadixSort::cMaxArraySize = BLOCK_SIZE_CUBE*4/(1<<CBITS);


oclRadixSort::oclRadixSort(oclContext& iContext)
: oclProgram(iContext, "oclRadixSort")
// buffers
, bfTempKey(iContext, "bfTempKey")
, bfTempVal(iContext, "bfTempVal")
, bfBlockScan(iContext, "bfBlockScan")
, bfBlockOffset(iContext, "bfBlockOffset")
, bfBlockSum(iContext, "bfBlockSum")
// kernels
, clBlockSort(*this)
, clBlockScan(*this)
, clBlockPrefix(*this)
, clReorder(*this)
{
    bfTempKey.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBlockSize);
    bfTempVal.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBlockSize);
    bfBlockScan.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBlockSize);
    bfBlockOffset.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBlockSize);
    bfBlockSum.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBlockSize);

    addSourceFile("sort/oclRadixSort.cl");

    exportKernel(clBlockSort);
    exportKernel(clBlockScan);
    exportKernel(clBlockPrefix);
    exportKernel(clReorder);
}

void oclRadixSort::fit(oclBuffer& iBuffer, size_t iElements) 
{
    if (iBuffer)
    {
        if (iBuffer.count<cl_uint>() < iElements)
        {
            iBuffer.resize<cl_uint>(iElements);
        }
    }
}

//
//
//

int oclRadixSort::compile()
{
    if (!oclProgram::compile())
    {
        return 0;
    }

    clBlockSort = createKernel("clBlockSort");
    KERNEL_VALIDATE(clBlockSort)
    clBlockScan = createKernel("clBlockScan");
    KERNEL_VALIDATE(clBlockScan)
    clBlockPrefix = createKernel("clBlockPrefix");
    KERNEL_VALIDATE(clBlockPrefix)
    clReorder = createKernel("clReorder");
    KERNEL_VALIDATE(clReorder)
    return 1;
}


int oclRadixSort::compute(oclDevice& iDevice, oclBuffer& bfKey, oclBuffer& bfVal, int iStartBit, int iEndBit)
{
    if (bfKey.dim(0) != bfVal.dim(0))
    {
        Log(ERR, this, __FILE__, __LINE__) << "key and value arrays are of different size ( " << bfKey.getMemObjectInfo<size_t>(CL_MEM_SIZE) << "," << bfVal.getMemObjectInfo<size_t>(CL_MEM_SIZE) << ")";
        return false;
    }

    if (bfKey.count<cl_uint>() >= cMaxArraySize)
    {
        Log(ERR, this, __FILE__, __LINE__) << "maximum sortable array size = " << cMaxArraySize;
        return false;
    } 

    if ((iEndBit - iStartBit) % cBits != 0)
    {
        Log(ERR, this, __FILE__, __LINE__) << "end bit(" << iEndBit << ") - start bit(" << iStartBit << ") must be divisible by 4";
        return false;
    } 

    size_t lBlockCount = ceil((float)bfKey.count<cl_uint>()/cBlockSize);
    fit(bfBlockScan, lBlockCount*(1<<cBits));
    fit(bfBlockOffset, lBlockCount*(1<<cBits));
    fit(bfBlockSum, cBlockSize);

    size_t lElementCount = bfKey.count<cl_uint>();
    fit(bfTempKey, lElementCount);
    fit(bfTempVal, lElementCount);

    size_t lGlobalSize = lBlockCount*cBlockSize;
    size_t lScanCount = lBlockCount*(1<<cBits)/4;
    size_t lScanSize = ceil((float)lScanCount/cBlockSize)*cBlockSize;

    for (int j=iStartBit; j<iEndBit; j+=cBits)
    {
        clSetKernelArg(clBlockSort, 0, sizeof(cl_mem), bfKey);
        clSetKernelArg(clBlockSort, 1, sizeof(cl_mem), bfTempKey);
        clSetKernelArg(clBlockSort, 2, sizeof(cl_mem), bfVal);
        clSetKernelArg(clBlockSort, 3, sizeof(cl_mem), bfTempVal);
        clSetKernelArg(clBlockSort, 4, sizeof(cl_uint), &j);
        clSetKernelArg(clBlockSort, 5, sizeof(cl_mem), bfBlockScan);
        clSetKernelArg(clBlockSort, 6, sizeof(cl_mem), bfBlockOffset);
        clSetKernelArg(clBlockSort, 7, sizeof(cl_uint), &lElementCount);
        sStatusCL = clEnqueueNDRangeKernel(iDevice.getQueue(), clBlockSort, 1, NULL, &lGlobalSize, &cBlockSize, 0, NULL, clBlockSort.getEvent());
        if (!oclSuccess("clEnqueueNDRangeKernel", this))
        {
            return false;
        } 

        clSetKernelArg(clBlockScan, 0, sizeof(cl_mem), bfBlockScan);
        clSetKernelArg(clBlockScan, 1, sizeof(cl_mem), bfBlockSum);
        clSetKernelArg(clBlockScan, 2, sizeof(cl_uint), &lScanCount);
        sStatusCL = clEnqueueNDRangeKernel(iDevice.getQueue(), clBlockScan, 1, NULL, &lScanSize, &cBlockSize, 0, NULL, clBlockScan.getEvent());
        if (!oclSuccess("clEnqueueNDRangeKernel", this))
        {
            return false;
        }

        clSetKernelArg(clBlockPrefix, 0, sizeof(cl_mem), bfBlockScan);
        clSetKernelArg(clBlockPrefix, 1, sizeof(cl_mem), bfBlockSum);
        clSetKernelArg(clBlockPrefix, 2, sizeof(cl_uint), &lScanCount);
        sStatusCL = clEnqueueNDRangeKernel(iDevice.getQueue(), clBlockPrefix, 1, NULL, &lScanSize, &cBlockSize, 0, NULL, clBlockPrefix.getEvent());
        if (!oclSuccess("clEnqueueNDRangeKernel", this))
        {
            return false;
        }

        clSetKernelArg(clReorder, 0, sizeof(cl_mem), bfTempKey);
        clSetKernelArg(clReorder, 1, sizeof(cl_mem), bfKey);
        clSetKernelArg(clReorder, 2, sizeof(cl_mem), bfTempVal);
        clSetKernelArg(clReorder, 3, sizeof(cl_mem), bfVal);
        clSetKernelArg(clReorder, 4, sizeof(cl_mem), bfBlockScan);
        clSetKernelArg(clReorder, 5, sizeof(cl_mem), bfBlockOffset);
        clSetKernelArg(clReorder, 6, sizeof(cl_uint), &j);
        clSetKernelArg(clReorder, 7, sizeof(cl_uint), &lElementCount);
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clReorder, 1, NULL, &lGlobalSize, &cBlockSize, 0, NULL, clReorder.getEvent());
        if (!oclSuccess("clEnqueueNDRangeKernel", this))
        {
            return false;
        }
    }
    return true;
};
