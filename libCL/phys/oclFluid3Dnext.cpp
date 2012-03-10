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
#include "oclFluid3Dnext.h"

char* oclFluid3Dnext::EVT_INTEGRATE = "OnIntegrate";

oclFluid3Dnext::oclFluid3Dnext(oclContext& iContext)
: oclProgram(iContext, "oclFluid3Dnext")
// buffers
, bfCell(iContext, "bfCell", oclBuffer::_uint)
, bfCellStart(iContext, "bfCellStart", oclBuffer::_uint)
, bfCellEnd(iContext, "bfCellEnd", oclBuffer::_uint)
, bfIndex(iContext, "bfIndex", oclBuffer::_uint)

, bfPressure(iContext, "bfPressure", oclBuffer::_float2)
, bfForce(iContext, "bfForce", oclBuffer::_float2)

, bfSortedState(iContext, "bfSortedState", oclBuffer::_float4)
, bfParams(iContext, "bfParams")
, bfPosition(0)
, bfBorder(0)

// kernels
, clHash(*this)
, clReorder(*this)
, clInitBounds(*this)
, clFindBounds(*this)

, clInitFluid(*this)

, clComputePressure(*this)
, clComputeForces(*this)
, clIntegrateForce(*this)

, clComputeVelocity(*this)

// programs
, mRadixSort(iContext)
// members
, mParticleCount(128*128)
, mIntegrateCb(0)

, mCellCount(256)
{
	bfCell.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
	bfCellStart.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mCellCount);
	bfCellEnd.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mCellCount);
	bfIndex.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);

	bfPressure.create<cl_float2>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
	bfForce.create<cl_float2>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);

	bfSortedState.create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);
	bfParams.create<Params>(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1, &mParams);

	bfBorder = new oclBuffer(iContext, "bfBorder");
	bfBorder->setOwner(this); 
	bfPosition = new oclBuffer(iContext, "bfPosition");
	bfPosition->setOwner(this); 
 
	bfBorder->create<cl_float2>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 2);
	bfPosition->create<cl_float4>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mParticleCount);

	addSourceFile("phys\\oclFluid3Dnext.cl");

	exportKernel(clReorder);
	exportKernel(clHash);
	exportKernel(clInitBounds);
	exportKernel(clFindBounds);

	exportKernel(clComputePressure);
	exportKernel(clComputeForces);
	exportKernel(clIntegrateForce);
}

oclFluid3Dnext::~oclFluid3Dnext()
{
	deleteBuffer(bfBorder);
	deleteBuffer(bfPosition);
}

void oclFluid3Dnext::setParticleCount(size_t iSize)
{
	// JSTIER make sure mParticleCount is a mutliple of cLocalSize
	mParticleCount = iSize;

	bfCell.resize<cl_uint>(mParticleCount);
	bfIndex.resize<cl_uint>(mParticleCount);

	bfPressure.resize<cl_float2>(mParticleCount);
	bfForce.resize<cl_float2>(mParticleCount);
	bfSortedState.resize<cl_float4>(mParticleCount);

	bfPosition->resize<cl_float4>(mParticleCount);
	if (bfPosition->getOwner<oclObject>() != this)
	{
		Log(WARN, this) << " resizing buffer " <<  bfPosition->getName() << " to " << mParticleCount;
	}

	bindBuffers();
}

size_t oclFluid3Dnext::getParticleCount()
{
	return mParticleCount;
}

int oclFluid3Dnext::setPositionBuffer(oclBuffer* iBuffer)
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

int oclFluid3Dnext::setBorderBuffer(oclBuffer* iBuffer)
{
	deleteBuffer(bfBorder);
	bfBorder = iBuffer;
	return bindBuffers();
}

//
//
//

oclBuffer* oclFluid3Dnext::getPositionBuffer()
{
	return bfPosition;
}
oclBuffer* oclFluid3Dnext::getBorderBuffer()
{
	return bfBorder;
}

oclBuffer& oclFluid3Dnext::getSortedPositionBuffer()
{
	return bfSortedState;
}
oclBuffer& oclFluid3Dnext::getIndexBuffer()
{
	return bfIndex;
}

oclBuffer& oclFluid3Dnext::getParamBuffer()
{
	return bfParams;
}


oclFluid3Dnext::Params& oclFluid3Dnext::getParameters()
{
	return *bfParams.ptr<oclFluid3Dnext::Params>();
};

//
//
//

int oclFluid3Dnext::bindBuffers()
{
    RETURN_ON_ERROR(0);

	clSetKernelArg(clHash, 0 ,sizeof(cl_mem), bfCell);
	clSetKernelArg(clHash, 1, sizeof(cl_mem), bfIndex);
	clSetKernelArg(clHash, 2, sizeof(cl_mem), *bfPosition);
	clSetKernelArg(clHash, 3, sizeof(cl_mem), bfParams);

	clSetKernelArg(clReorder, 0, sizeof(cl_mem), bfIndex);
	clSetKernelArg(clReorder, 1, sizeof(cl_mem), *bfPosition);
	clSetKernelArg(clReorder, 2, sizeof(cl_mem), bfSortedState);

	clSetKernelArg(clInitBounds, 0, sizeof(cl_mem), bfCellStart);
	clSetKernelArg(clInitBounds, 1, sizeof(cl_mem), bfCellEnd);

	clSetKernelArg(clFindBounds, 0, sizeof(cl_mem), bfCellStart); 
	clSetKernelArg(clFindBounds, 1, sizeof(cl_mem), bfCellEnd);
	clSetKernelArg(clFindBounds, 2, sizeof(cl_mem), bfCell);
	clSetKernelArg(clFindBounds, 3, (256 + 1) * sizeof(cl_uint), 0);

	//
	//
	//

	clSetKernelArg(clComputePressure, 0, sizeof(cl_mem), bfSortedState);
	clSetKernelArg(clComputePressure, 1, sizeof(cl_mem), bfPressure);
	clSetKernelArg(clComputePressure, 2, sizeof(cl_mem), bfCellStart);
	clSetKernelArg(clComputePressure, 3, sizeof(cl_mem), bfCellEnd);
	clSetKernelArg(clComputePressure, 4, sizeof(cl_mem), bfParams);

	clSetKernelArg(clComputeForces, 0, sizeof(cl_mem), bfSortedState);
	clSetKernelArg(clComputeForces, 1, sizeof(cl_mem), bfPressure);
	clSetKernelArg(clComputeForces, 2, sizeof(cl_mem), bfForce);
	clSetKernelArg(clComputeForces, 3, sizeof(cl_mem), bfCellStart);
	clSetKernelArg(clComputeForces, 4, sizeof(cl_mem), bfCellEnd);
	clSetKernelArg(clComputeForces, 5, sizeof(cl_mem), bfParams);

	clSetKernelArg(clIntegrateForce, 0, sizeof(cl_mem), *bfPosition);
	clSetKernelArg(clIntegrateForce, 1, sizeof(cl_mem), bfForce);
	clSetKernelArg(clIntegrateForce, 2, sizeof(cl_mem), bfIndex);
	clSetKernelArg(clIntegrateForce, 3, sizeof(cl_mem), bfParams);

	/*
	clSetKernelArg(clIntegrateForce, 0, sizeof(cl_mem), *bfState);
	clSetKernelArg(clIntegrateForce, 1, sizeof(cl_mem), *bfForce);
	clSetKernelArg(clIntegrateForce, 2, sizeof(cl_mem), bfParams);

	clSetKernelArg(clIntegrateVelocity, 0, sizeof(cl_mem), *bfPosition);
	clSetKernelArg(clIntegrateVelocity, 1, sizeof(cl_mem), *bfState);
	clSetKernelArg(clIntegrateVelocity, 2, sizeof(cl_mem), bfParams);

	clSetKernelArg(clCalculateDensity, 0, sizeof(cl_mem), bfSortedState);
	clSetKernelArg(clCalculateDensity, 2, sizeof(cl_mem), bfCellStart);
	clSetKernelArg(clCalculateDensity, 3, sizeof(cl_mem), bfCellEnd);
	clSetKernelArg(clCalculateDensity, 4, sizeof(cl_mem), bfParams);

	clSetKernelArg(clCalculateForces, 0, sizeof(cl_mem), bfSortedState);
	clSetKernelArg(clCalculateForces, 1, sizeof(cl_mem), bfSortedState);
	clSetKernelArg(clCalculateForces, 2, sizeof(cl_mem), *bfForce);
	clSetKernelArg(clCalculateForces, 3, sizeof(cl_mem), bfIndex);
	clSetKernelArg(clCalculateForces, 4, sizeof(cl_mem), bfCellStart);
	clSetKernelArg(clCalculateForces, 5, sizeof(cl_mem), bfCellEnd);
	clSetKernelArg(clCalculateForces, 6, sizeof(cl_mem), bfParams);
		*/
	//
	//
	//

	return 1;
}

int oclFluid3Dnext::compile()
{
	clHash = 0;
	clReorder = 0;
	clInitBounds = 0;
	clFindBounds = 0;

	clInitFluid = 0;

	clComputePressure = 0;
	clComputeForces = 0;
	clIntegrateForce = 0;
	clComputeVelocity = 0;

	if (!mRadixSort.compile())
	{
		return 0;
	}

	if (!oclProgram::compile())
	{
		return 0;
	}

	clHash = createKernel("clHash");
	KERNEL_VALIDATE(clHash)
	clReorder = createKernel("clReorder");
	KERNEL_VALIDATE(clReorder)
	clInitBounds = createKernel("clInitBounds");
	KERNEL_VALIDATE(clInitBounds)
	clFindBounds = createKernel("clFindBounds");
	KERNEL_VALIDATE(clFindBounds)

	clInitFluid = createKernel("clInitFluid");
	KERNEL_VALIDATE(clInitFluid)

	clComputePressure = createKernel("clComputePressure");
	KERNEL_VALIDATE(clComputePressure)
	clComputeForces = createKernel("clComputeForces");
	KERNEL_VALIDATE(clComputeForces)
	clIntegrateForce = createKernel("clIntegrateForce");
	KERNEL_VALIDATE(clIntegrateForce)
	clComputeVelocity = createKernel("clComputeVelocity");
	KERNEL_VALIDATE(clComputeVelocity)

    // init fluid parameters
	clSetKernelArg(clInitFluid, 0, sizeof(cl_mem), bfParams);
	clEnqueueTask(mContext.getDevice(0), clInitFluid, 0, NULL, clInitFluid.getEvent());
	bfParams.map(CL_MAP_READ);

    Params& lParams = getParameters();
    mCellCount = lParams.cellCountX*lParams.cellCountY;
	bfCellStart.resize<cl_uint>(mCellCount);
	bfCellEnd.resize<cl_uint>(mCellCount);

	return bindBuffers();
}

void oclFluid3Dnext::addEventHandler(srtEvent& iEvent)
{
    oclProgram::addEventHandler(iEvent);
	mIntegrateCb = getEventHandler(EVT_INTEGRATE);
}

int oclFluid3Dnext::compute(oclDevice& iDevice)
{
    RETURN_ON_ERROR(0);

	sStatusCL = clEnqueueNDRangeKernel(iDevice, clHash, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clHash.getEvent());
	ENQUEUE_VALIDATE

	// sort
	if (!mRadixSort.compute(iDevice, bfCell, bfIndex, 0, 24))
	{
		return false;
	}

	sStatusCL = clEnqueueNDRangeKernel(iDevice, clReorder, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clReorder.getEvent());
	ENQUEUE_VALIDATE

	sStatusCL = clEnqueueNDRangeKernel(iDevice, clInitBounds, 1, NULL, &mCellCount, &cLocalSize, 0, NULL, clInitBounds.getEvent());
	ENQUEUE_VALIDATE
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clFindBounds, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clFindBounds.getEvent());
	ENQUEUE_VALIDATE
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clComputePressure, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clComputePressure.getEvent());
	ENQUEUE_VALIDATE
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clComputeForces, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clComputeForces.getEvent());
	ENQUEUE_VALIDATE
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clIntegrateForce, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clIntegrateForce.getEvent());
	ENQUEUE_VALIDATE

/*

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
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clApplyGravity, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clApplyGravity.getEvent());
		ENQUEUE_VALIDATE
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clIntegrateVelocity, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clIntegrateVelocity.getEvent());
		ENQUEUE_VALIDATE
	}
	*/
	return true;
}

