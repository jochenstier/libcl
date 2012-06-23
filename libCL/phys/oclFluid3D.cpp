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
#include "oclFluid3D.h"

const size_t oclFluid3D::cLocalSize = 256;
const size_t oclFluid3D::cBucketCount = 16777216;

char* oclFluid3D::EVT_INTEGRATE = "OnIntegrate";

oclFluid3D::oclFluid3D(oclContext& iContext)
: oclProgram(iContext, "oclFluid3D")
// buffers
, bfCell(iContext, "bfCell")
, bfCellStart(iContext, "bfCellStart")
, bfCellEnd(iContext, "bfCellEnd")
, bfIndex(iContext, "bfIndex")
, bfSortedPosition(iContext, "bfSortedPosition")
, bfSortedVelocity(iContext, "bfSortedVelocity")
, bfParams(iContext, "bfParams")
, bfPosition(0)
, bfVelocity(0)
, bfForce(0)
// kernels
, clIntegrateForce(*this)
, clIntegrateVelocity(*this)
, clHash(*this)
, clReorder(*this)
, clInitBounds(*this)
, clFindBounds(*this)
, clCalculateDensity(*this)
, clCalculateForces(*this)
, clInitFluid(*this)
, clClipBox(*this)
, clGravity(*this)
// programs
, mRadixSort(iContext)
// members
, mParticleCount(cLocalSize)
, mIntegrateCb(0)
{
    bfCell.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
    bfCellStart.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBucketCount);
    bfCellEnd.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBucketCount);
    bfIndex.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
    bfSortedPosition.create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
    bfSortedVelocity.create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
    bfParams.create<Params>(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1, &mParams);

   /*
    oclBuffer* ll = new oclBuffer(iContext, "dsdsdsdsd");
    for (int i=0; i<100; i++)
    {
        ll->create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cBucketCount);
        delete ll;
        ll = new oclBuffer(iContext, "dsdsdsdsd");
    }
    */
    

    bfPosition = new oclBuffer(iContext, "bfPosition");
    bfPosition->setOwner(this); 
    bfVelocity = new oclBuffer(iContext, "bfVelocity");
    bfVelocity->setOwner(this);
    bfForce = new oclBuffer(iContext, "bfForce");
    bfForce->setOwner(this); 
 
    bfPosition->create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
    bfVelocity->create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
    bfForce->create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);

    addSourceFile("phys/oclFluid3D.cl");

    exportKernel(clClipBox);
    exportKernel(clGravity);
    exportKernel(clIntegrateForce);
    exportKernel(clIntegrateVelocity);
}

oclFluid3D::~oclFluid3D()
{
    deleteBuffer(bfPosition);
    deleteBuffer(bfVelocity);
    deleteBuffer(bfForce);
}

void oclFluid3D::setParticleCount(size_t iSize)
{
    // JSTIER make sure mParticleCount is a mutliple of cLocalSize

    mParticleCount = iSize;

    bfCell.resize<cl_uint>(mParticleCount);
    bfCellStart.resize<cl_uint>(cBucketCount);
    bfCellEnd.resize<cl_uint>(cBucketCount);
    bfIndex.resize<cl_uint>(mParticleCount);
    bfSortedPosition.resize<cl_float4>(mParticleCount);
    bfSortedVelocity.resize<cl_float4>(mParticleCount);

    bfPosition->resize<cl_float4>(mParticleCount);
    if (bfPosition->getOwner<oclObject>() != this)
    {
        Log(WARN, this) << " resizing buffer " <<  bfPosition->getName() << " to " << mParticleCount;
    }

    bfVelocity->resize<cl_float4>(mParticleCount);
    if (bfVelocity->getOwner<oclObject>() != this)
    {
        Log(WARN, this) << " resizing buffer " <<  bfVelocity->getName() << " to " << mParticleCount;
    }

    bfForce->resize<cl_float4>(mParticleCount);
    if (bfForce->getOwner<oclObject>() != this)
    {
        Log(WARN, this) << " resizing buffer " <<  bfForce->getName() << " to " << mParticleCount;
    }

    bindBuffers();
}

size_t oclFluid3D::getParticleCount()
{
    return mParticleCount;
}

int oclFluid3D::setPositionBuffer(oclBuffer* iBuffer)
{
    if (iBuffer->count<cl_float4>() != mParticleCount)
    {
        iBuffer->resize<cl_float4>(mParticleCount);
        Log(WARN, this) << " resizing buffer " <<  iBuffer->getName() << " to " << mParticleCount;
    }
    deleteBuffer(bfPosition);
    bfPosition = iBuffer;
    return bindBuffers();
}

int oclFluid3D::setVelocityBuffer(oclBuffer* iBuffer)
{
    if (iBuffer->count<cl_float4>() < mParticleCount)
    {
        iBuffer->resize<cl_float4>(mParticleCount);
        Log(WARN, this) << " libCL is resizing buffer " <<  iBuffer->getName() << " to " << mParticleCount;
    }
    deleteBuffer(bfVelocity);
    bfVelocity = iBuffer;
    return bindBuffers();
}

int oclFluid3D::setForceBuffer(oclBuffer* iBuffer)
{
    if (iBuffer->count<cl_float4>() < mParticleCount)
    {
        iBuffer->resize<cl_float4>(mParticleCount);
        Log(WARN, this) << " libCL is resizing buffer " <<  iBuffer->getName() << " to " << mParticleCount;
    }
    deleteBuffer(bfForce);
    bfForce = iBuffer;
    return bindBuffers();
}

//
//
//

oclBuffer* oclFluid3D::getPositionBuffer()
{
    return bfPosition;
}
oclBuffer* oclFluid3D::getVelocityBuffer()
{
    return bfVelocity;
}

oclBuffer* oclFluid3D::getForceBuffer()
{
    return bfForce;
}

oclBuffer& oclFluid3D::getSortedPositionBuffer()
{
    return bfSortedPosition;
}
oclBuffer& oclFluid3D::getSortedVelocityBuffer()
{
    return bfSortedVelocity;
}
oclBuffer& oclFluid3D::getIndexBuffer()
{
    return bfIndex;
}

oclBuffer& oclFluid3D::getParamBuffer()
{
    return bfParams;
}


oclFluid3D::Params& oclFluid3D::getParameters()
{
    return *bfParams.ptr<oclFluid3D::Params>();
};

//
//
//

int oclFluid3D::bindBuffers()
{
    clSetKernelArg(clIntegrateForce, 0, sizeof(cl_mem), *bfVelocity);
    clSetKernelArg(clIntegrateForce, 1, sizeof(cl_mem), *bfForce);
    clSetKernelArg(clIntegrateForce, 2, sizeof(cl_mem), bfParams);

    clSetKernelArg(clIntegrateVelocity, 0, sizeof(cl_mem), *bfPosition);
    clSetKernelArg(clIntegrateVelocity, 1, sizeof(cl_mem), *bfVelocity);
    clSetKernelArg(clIntegrateVelocity, 2, sizeof(cl_mem), bfParams);

    clSetKernelArg(clHash, 0 ,sizeof(cl_mem), bfCell);
    clSetKernelArg(clHash, 1, sizeof(cl_mem), bfIndex);
    clSetKernelArg(clHash, 2, sizeof(cl_mem), *bfPosition);
    clSetKernelArg(clHash, 3, sizeof(cl_mem), bfParams);

    clSetKernelArg(clReorder, 0, sizeof(cl_mem), bfIndex);
    clSetKernelArg(clReorder, 1, sizeof(cl_mem), *bfPosition);
    clSetKernelArg(clReorder, 2, sizeof(cl_mem), *bfVelocity);
    clSetKernelArg(clReorder, 3, sizeof(cl_mem), bfSortedPosition);
    clSetKernelArg(clReorder, 4, sizeof(cl_mem), bfSortedVelocity);

    clSetKernelArg(clInitBounds, 0, sizeof(cl_mem), bfCellStart);
    clSetKernelArg(clInitBounds, 1, sizeof(cl_mem), bfCellEnd);

    clSetKernelArg(clFindBounds, 0, sizeof(cl_mem), bfCellStart); 
    clSetKernelArg(clFindBounds, 1, sizeof(cl_mem), bfCellEnd);
    clSetKernelArg(clFindBounds, 2, sizeof(cl_mem), bfCell);
    clSetKernelArg(clFindBounds, 3, (256 + 1) * sizeof(cl_uint), 0);

    clSetKernelArg(clCalculateDensity, 0, sizeof(cl_mem), bfSortedPosition);
    clSetKernelArg(clCalculateDensity, 1, sizeof(cl_mem), bfSortedVelocity);
    clSetKernelArg(clCalculateDensity, 2, sizeof(cl_mem), bfCellStart);
    clSetKernelArg(clCalculateDensity, 3, sizeof(cl_mem), bfCellEnd);
    clSetKernelArg(clCalculateDensity, 4, sizeof(cl_mem), bfParams);

    clSetKernelArg(clCalculateForces, 0, sizeof(cl_mem), bfSortedPosition);
    clSetKernelArg(clCalculateForces, 1, sizeof(cl_mem), bfSortedVelocity);
    clSetKernelArg(clCalculateForces, 2, sizeof(cl_mem), *bfForce);
    clSetKernelArg(clCalculateForces, 3, sizeof(cl_mem), bfIndex);
    clSetKernelArg(clCalculateForces, 4, sizeof(cl_mem), bfCellStart);
    clSetKernelArg(clCalculateForces, 5, sizeof(cl_mem), bfCellEnd);
    clSetKernelArg(clCalculateForces, 6, sizeof(cl_mem), bfParams);

    cl_float4 lGravity = { 0, 0, 1, 0 };
    clSetKernelArg(clGravity, 0, sizeof(cl_mem), *bfForce);
    clSetKernelArg(clGravity, 1, sizeof(cl_float4), &lGravity);

    cl_float4 lMin = { -1, -1, -1, 0 };
    cl_float4 lMax = {  1,  1,  1, 0 };
    clSetKernelArg(clClipBox, 0, sizeof(cl_mem), bfSortedPosition);
    clSetKernelArg(clClipBox, 1, sizeof(cl_mem), bfSortedVelocity);
    clSetKernelArg(clClipBox, 2, sizeof(cl_mem), *bfForce);
    clSetKernelArg(clClipBox, 3, sizeof(cl_mem), bfIndex);
    clSetKernelArg(clClipBox, 4, sizeof(cl_mem), bfParams);
    clSetKernelArg(clClipBox, 5, sizeof(cl_float4), &lMin);
    clSetKernelArg(clClipBox, 6, sizeof(cl_float4), &lMax);
    return 1;
}


int oclFluid3D::compile()
{
    clInitFluid = 0;
    clIntegrateForce = 0;
    clIntegrateVelocity = 0;
    clHash = 0;
    clReorder = 0;
    clInitBounds = 0;

    if (!mRadixSort.compile())
    {
        return 0;
    }

    if (!oclProgram::compile())
    {
        return 0;
    }

    clInitFluid = createKernel("clInitFluid");
    KERNEL_VALIDATE(clInitFluid)
    clIntegrateForce = createKernel("clIntegrateForce");
    KERNEL_VALIDATE(clIntegrateForce)
    clIntegrateVelocity = createKernel("clIntegrateVelocity");
    KERNEL_VALIDATE(clIntegrateVelocity)
    clHash = createKernel("clHash");
    KERNEL_VALIDATE(clHash)
    clReorder = createKernel("clReorder");
    KERNEL_VALIDATE(clReorder)
    clInitBounds = createKernel("clInitBounds");
    KERNEL_VALIDATE(clInitBounds)
    clFindBounds = createKernel("clFindBounds");
    KERNEL_VALIDATE(clFindBounds)
    clCalculateDensity = createKernel("clCalculateDensity");
    KERNEL_VALIDATE(clCalculateDensity)
    clCalculateForces = createKernel("clCalculateForces");
    KERNEL_VALIDATE(clCalculateForces)
    clGravity = createKernel("clGravity");
    KERNEL_VALIDATE(clGravity)
    clClipBox = createKernel("clClipBox");
    KERNEL_VALIDATE(clClipBox)

    // init fluid parameters
    clSetKernelArg(clInitFluid, 0, sizeof(cl_mem), bfParams);
    clEnqueueTask(mContext.getDevice(0), clInitFluid, 0, NULL, clInitFluid.getEvent());
    bfParams.map(CL_MAP_READ);

    return bindBuffers();
}

void oclFluid3D::addEventHandler(srtEvent& iEvent)
{
    oclProgram::addEventHandler(iEvent);
    mIntegrateCb = getEventHandler(EVT_INTEGRATE);
}

int oclFluid3D::compute(oclDevice& iDevice)
{
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clHash, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clHash.getEvent());
    ENQUEUE_VALIDATE

     // sort
    if (!mRadixSort.compute(iDevice, bfCell, bfIndex, 0, 24))
    {
        return false;
    }

    sStatusCL = clEnqueueNDRangeKernel(iDevice, clReorder, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clReorder.getEvent());
    ENQUEUE_VALIDATE

    sStatusCL = clEnqueueNDRangeKernel(iDevice, clInitBounds, 1, NULL, &cBucketCount, &cLocalSize, 0, NULL, clInitBounds.getEvent());
    ENQUEUE_VALIDATE
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clFindBounds, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clFindBounds.getEvent());
    ENQUEUE_VALIDATE

    sStatusCL = clEnqueueNDRangeKernel(iDevice, clCalculateDensity, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clCalculateDensity.getEvent());
    ENQUEUE_VALIDATE

    sStatusCL = clEnqueueNDRangeKernel(iDevice, clCalculateForces, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clCalculateForces.getEvent());
    ENQUEUE_VALIDATE

    // resolve soft constraints
    if (mIntegrateCb)
    {
        (*mIntegrateCb)(*this);
    }
    else
    {
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clClipBox, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clClipBox.getEvent());
        ENQUEUE_VALIDATE
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clIntegrateForce, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clIntegrateForce.getEvent());
        ENQUEUE_VALIDATE
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clGravity, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clGravity.getEvent());
        ENQUEUE_VALIDATE
        sStatusCL = clEnqueueNDRangeKernel(iDevice, clIntegrateVelocity, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clIntegrateVelocity.getEvent());
        ENQUEUE_VALIDATE
    }

    return true;
}

