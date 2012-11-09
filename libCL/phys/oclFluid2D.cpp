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
#include "oclFluid2D.h"

char* oclFluid2D::EVT_INTEGRATE = "OnIntegrate";

oclFluid2D::oclFluid2D(oclContext& iContext)
: oclProgram(iContext, "oclFluid2D")
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
, clInitPressure(*this)

, clComputePressure(*this)
, clComputeForces(*this)
, clIntegrateForce(*this)
, clIntegrateVelocity(*this)

, clCollideBVH(*this)

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

	addSourceFile("phys\\oclFluid2D.cl");

	exportKernel(clReorder);
	exportKernel(clHash);
	exportKernel(clInitBounds);
	exportKernel(clFindBounds);
	exportKernel(clInitPressure);

	exportKernel(clComputePressure);
	exportKernel(clComputeForces);
	exportKernel(clIntegrateForce);
	exportKernel(clCollideBVH);
}

oclFluid2D::~oclFluid2D()
{
	deleteBuffer(bfBorder);
	deleteBuffer(bfPosition);
}

void oclFluid2D::setParticleCount(size_t iSize)
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

size_t oclFluid2D::getParticleCount()
{
	return mParticleCount;
}

int oclFluid2D::setPositionBuffer(oclBuffer* iBuffer)
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

int oclFluid2D::setBorderBuffer(oclBuffer* iBuffer)
{
	deleteBuffer(bfBorder);
	bfBorder = iBuffer;
	return bindBuffers();
}

//
//
//

oclBuffer* oclFluid2D::getPositionBuffer()
{
	return bfPosition;
}
oclBuffer* oclFluid2D::getBorderBuffer()
{
	return bfBorder;
}

oclBuffer& oclFluid2D::getSortedPositionBuffer()
{
	return bfSortedState;
}
oclBuffer& oclFluid2D::getIndexBuffer()
{
	return bfIndex;
}

oclBuffer& oclFluid2D::getParamBuffer()
{
	return bfParams;
}


oclFluid2D::Params& oclFluid2D::getParameters()
{
	return *bfParams.ptr<oclFluid2D::Params>();
};

//
//
//

int oclFluid2D::bindBuffers()
{
    RETURN_ON_ERROR(0);

	clSetKernelArg(clInitPressure, 0, sizeof(cl_mem), bfPressure);
	clSetKernelArg(clInitPressure, 1, sizeof(cl_mem), bfParams);
	sStatusCL = clEnqueueNDRangeKernel(mContext.getDevice(0), clInitPressure, 1, NULL, &mParticleCount, 0, 0, NULL, clInitPressure.getEvent());
	ENQUEUE_VALIDATE

    //
    //
    //

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

	clSetKernelArg(clIntegrateVelocity, 0, sizeof(cl_mem), *bfPosition);
	clSetKernelArg(clIntegrateVelocity, 1, sizeof(cl_mem), bfParams);
	return 1;
}

int oclFluid2D::compile()
{
	clHash = 0;
	clReorder = 0;
	clInitBounds = 0;
	clFindBounds = 0;

	clInitFluid = 0;
	clInitPressure = 0;

	clComputePressure = 0;
	clComputeForces = 0;
	clIntegrateForce = 0;
	clIntegrateVelocity = 0;

	clCollideBVH = 0;

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
	clInitPressure = createKernel("clInitPressure");
	KERNEL_VALIDATE(clInitPressure)

	clComputePressure = createKernel("clComputePressure");
	KERNEL_VALIDATE(clComputePressure)
	clComputeForces = createKernel("clComputeForces");
	KERNEL_VALIDATE(clComputeForces)
	clIntegrateForce = createKernel("clIntegrateForce");
	KERNEL_VALIDATE(clIntegrateForce)
	clIntegrateVelocity = createKernel("clIntegrateVelocity");
	KERNEL_VALIDATE(clIntegrateVelocity)
	clCollideBVH = createKernel("clCollideBVH");
	KERNEL_VALIDATE(clCollideBVH)

    // init fluid parameters
	clSetKernelArg(clInitFluid, 0, sizeof(cl_mem), bfParams);
	clEnqueueTask(mContext.getDevice(0), clInitFluid, 0, NULL, clInitFluid.getEvent());
	bfParams.map(CL_MAP_READ);

    Params& lParams = getParameters();
    mCellCount = lParams.cellCountX*lParams.cellCountY;
    mCellCount = (mCellCount/cLocalSize+1)*cLocalSize;
	bfCellStart.resize<cl_uint>(mCellCount);
	bfCellEnd.resize<cl_uint>(mCellCount);

	return bindBuffers();
}

void oclFluid2D::addEventHandler(srtEvent& iEvent)
{
    oclProgram::addEventHandler(iEvent);
	mIntegrateCb = getEventHandler(EVT_INTEGRATE);
}

int oclFluid2D::compute(oclDevice& iDevice)
{
    RETURN_ON_ERROR(0);

	sStatusCL = clEnqueueNDRangeKernel(iDevice, clHash, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clHash.getEvent());
	ENQUEUE_VALIDATE

	// sort
	if (!mRadixSort.compute(iDevice, bfCell, bfIndex, 0, 24))
	{
		return false;
	}
/*
	if (bfCell.map(CL_MAP_READ))
	{
        int lDim = bfCell.dim(0)/sizeof(cl_uint);
		cl_uint* lPtr = bfCell.ptr<cl_uint>();

		for (int i=0; i<lDim; i++)
		{
			Log(ERR) << "cell" << lPtr[i];
		}
	}
*/

	sStatusCL = clEnqueueNDRangeKernel(iDevice, clReorder, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clReorder.getEvent());
	ENQUEUE_VALIDATE

	sStatusCL = clEnqueueNDRangeKernel(iDevice, clInitBounds, 1, NULL, &mCellCount, &cLocalSize, 0, NULL, clInitBounds.getEvent());
	ENQUEUE_VALIDATE
 	sStatusCL = clEnqueueNDRangeKernel(iDevice, clFindBounds, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clFindBounds.getEvent());
	ENQUEUE_VALIDATE

		
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clComputePressure, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clComputePressure.getEvent());
	ENQUEUE_VALIDATE

	if (bfPressure.map(CL_MAP_READ))
	{
        int lDim = bfPressure.dim(0)/sizeof(cl_float2);
		cl_float2* lPtr = bfPressure.ptr<cl_float2>();
		for (int i=0; i<lDim; i++)
		{
			Log(INFO) << i << " pressure " << lPtr[i];
		}
		bfPressure.unmap();
	}

   
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clComputeForces, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clComputeForces.getEvent());
	ENQUEUE_VALIDATE

	sStatusCL = clEnqueueNDRangeKernel(iDevice, clIntegrateForce, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clIntegrateForce.getEvent());
	ENQUEUE_VALIDATE

	if (mIntegrateCb)
	{
		(*mIntegrateCb)(*this);
	}

   	sStatusCL = clEnqueueNDRangeKernel(iDevice, clIntegrateVelocity, 1, NULL, &mParticleCount, &cLocalSize, 0, NULL, clIntegrateVelocity.getEvent());
	ENQUEUE_VALIDATE 
 /**/
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

