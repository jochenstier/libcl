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

#include "oclBvhLines.h"

oclBvhLines::oclBvhLines(oclContext& iContext)
: oclProgram(iContext, "oclBvhLines")
// buffers
, bfMortonKey(iContext, "bfMortonKey", oclBuffer::_uint)
, bfMortonVal(iContext, "bfMortonVal", oclBuffer::_uint)
, bfBvhRoot(iContext, "bfBvhRoot")
, bfBvhNode(iContext, "bfBvhNode")
// kernels
, clMorton(*this)
, clCreateNodes(*this)
, clLinkNodes(*this)
, clCreateLeaves(*this)
, clComputeAABBs(*this)
// programs
, mRadixSort(iContext)
// members
, mRootNode(0)
{
	bfMortonKey.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 256);
	bfMortonVal.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 256);
	bfBvhNode.create<cl_char>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
	bfBvhRoot.create<cl_uint>(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 1, &mRootNode);

	addSourceFile("geom\\oclBvhLines.cl");

	exportKernel(clMorton);
	exportKernel(clCreateNodes);
	exportKernel(clLinkNodes);
	exportKernel(clCreateLeaves);
	exportKernel(clComputeAABBs);
}

oclBvhLines::~oclBvhLines()
{
}


int oclBvhLines::compile()
{
	clMorton = 0;
	clCreateNodes = 0;
	clLinkNodes = 0;
	clCreateLeaves = 0;
	clComputeAABBs = 0;

	if (!mRadixSort.compile())
	{
		return 0;
	}

	if (!oclProgram::compile())
	{
		return 0;
	}

	clMorton = createKernel("clMorton");
	KERNEL_VALIDATE(clMorton)
	clCreateNodes = createKernel("clCreateNodes");
	KERNEL_VALIDATE(clCreateNodes)
	clLinkNodes = createKernel("clLinkNodes");
	KERNEL_VALIDATE(clLinkNodes)
	clCreateLeaves = createKernel("clCreateLeaves");
	KERNEL_VALIDATE(clCreateLeaves)
	clComputeAABBs = createKernel("clComputeAABBs");
	KERNEL_VALIDATE(clComputeAABBs)
	return 1;
}


int oclBvhLines::compute(oclDevice& iDevice, oclBuffer& bfVertex)
{
	size_t lVertices = bfVertex.count<cl_float2>();
	size_t lLines = lVertices/2;

	if (bfMortonKey.count<cl_uint>() != lLines)
	{
		bfMortonKey.resize<cl_uint>(lLines);
	}
	if (bfMortonVal.count<cl_uint>() != lLines)
	{
		bfMortonVal.resize<cl_uint>(lLines);
	}
	if (bfBvhNode.count<BVHNode>() != 2*lLines-1)
	{
		bfBvhNode.resize<BVHNode>(2*lLines-1);
	}


	//
	// Compute morton curve
	//	 
	clSetKernelArg(clMorton, 0, sizeof(cl_mem), bfVertex);
	clSetKernelArg(clMorton, 1, sizeof(cl_mem), bfMortonKey);
	clSetKernelArg(clMorton, 2, sizeof(cl_mem), bfMortonVal);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clMorton, 1, NULL, &lLines, NULL, 0, NULL, clMorton.getEvent());
	ENQUEUE_VALIDATE

	// 
	// Sort morton curve
	//	
	if (!mRadixSort.compute(iDevice, bfMortonKey, bfMortonVal, 0, 32))
	{
		return 0;
	}

	//
	// Create BVH
	//	
	size_t lGlobalSize = lLines-1;

	clSetKernelArg(clCreateNodes, 0, sizeof(cl_mem), bfMortonKey);
	clSetKernelArg(clCreateNodes, 1, sizeof(cl_mem), bfMortonVal);
	clSetKernelArg(clCreateNodes, 2, sizeof(cl_mem), bfBvhNode);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clCreateNodes, 1, NULL, &lGlobalSize, NULL, 0, NULL, clCreateNodes.getEvent());
	ENQUEUE_VALIDATE

	clSetKernelArg(clLinkNodes, 0, sizeof(cl_mem), bfMortonKey);
	clSetKernelArg(clLinkNodes, 1, sizeof(cl_mem), bfBvhNode);
	clSetKernelArg(clLinkNodes, 2, sizeof(cl_mem), bfBvhRoot);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clLinkNodes, 1, NULL, &lGlobalSize, NULL, 0, NULL, clLinkNodes.getEvent());
	ENQUEUE_VALIDATE

	clSetKernelArg(clCreateLeaves, 0, sizeof(cl_mem), bfMortonVal);
	clSetKernelArg(clCreateLeaves, 1, sizeof(cl_mem), bfBvhNode);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clCreateLeaves, 1, NULL, &lLines, NULL, 0, NULL, clCreateLeaves.getEvent());
	ENQUEUE_VALIDATE

	clSetKernelArg(clComputeAABBs, 0, sizeof(cl_mem), bfVertex);
	clSetKernelArg(clComputeAABBs, 1, sizeof(cl_mem), bfBvhNode);
	clSetKernelArg(clComputeAABBs, 2, sizeof(cl_mem), bfBvhRoot);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clComputeAABBs, 1, NULL, &lLines, NULL, 0, NULL, clComputeAABBs.getEvent());
	ENQUEUE_VALIDATE
	
	bfBvhRoot.map(CL_MAP_READ);
	bfBvhRoot.unmap();

	return true;
}
//

cl_uint oclBvhLines::getRootNode()
{
	return mRootNode;
};

oclBuffer& oclBvhLines::getNodeBuffer()
{
	return bfBvhNode;
};

cl_uint oclBvhLines::getNodeCount()
{
	return (int)bfBvhNode.count<BVHNode>();
};
