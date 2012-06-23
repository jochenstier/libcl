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

#include "oclTangent.h"

oclTangent::oclTangent(oclContext& iContext)
: oclProgram(iContext, "oclTangent")
// kernels
, clTangent(*this)
, clLineConv(*this)
{
    addSourceFile("filter/oclTangent.cl");

    exportKernel(clTangent);
    exportKernel(clLineConv);
}

//
//
//

int oclTangent::compile()
{
    clTangent = 0;
    clLineConv = 0;
    if (!oclProgram::compile())
    {
        return 0;
    }
    clTangent = createKernel("clTangent");
    KERNEL_VALIDATE(clTangent)
    clLineConv = createKernel("clLineConv");
    KERNEL_VALIDATE(clLineConv)
    return 1;
}

//
//
//

int oclTangent::compute(oclDevice& iDevice, oclImage2D& bfDx, oclImage2D& bfDy, oclImage2D& bfDest)
{
    cl_uint lIw = bfDest.getImageInfo<size_t>(CL_IMAGE_WIDTH);
    cl_uint lIh = bfDest.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
    size_t lGlobalSize[2];
    size_t lLocalSize[2];
    clTangent.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

    clSetKernelArg(clTangent, 0, sizeof(cl_mem), bfDx);
    clSetKernelArg(clTangent, 1, sizeof(cl_mem), bfDy);
    clSetKernelArg(clTangent, 2, sizeof(cl_mem), bfDest);
    clSetKernelArg(clTangent, 3, sizeof(cl_uint), &lIw);
    clSetKernelArg(clTangent, 4, sizeof(cl_uint), &lIh);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clTangent, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clTangent.getEvent());
    ENQUEUE_VALIDATE
    return true;
}

int oclTangent::lineConv(oclDevice& iDevice, oclImage2D& bfVector, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iDepth)
{
    cl_uint lIw = bfDest.getImageInfo<size_t>(CL_IMAGE_WIDTH);
    cl_uint lIh = bfDest.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
    size_t lGlobalSize[2];
    size_t lLocalSize[2];
    clLineConv.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

    clSetKernelArg(clLineConv, 0, sizeof(cl_mem), bfVector);
    clSetKernelArg(clLineConv, 1, sizeof(cl_mem), bfSrce);
    clSetKernelArg(clLineConv, 2, sizeof(cl_mem), bfDest);
    clSetKernelArg(clLineConv, 3, sizeof(cl_uint), &iDepth);
    clSetKernelArg(clLineConv, 4, sizeof(cl_uint), &lIw);
    clSetKernelArg(clLineConv, 5, sizeof(cl_uint), &lIh);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, clLineConv, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clLineConv.getEvent());
    ENQUEUE_VALIDATE
    return true;
}
