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

#include "oclSobel.h"


oclSobel::oclSobel(oclContext& iContext)
: oclProgram(iContext, "oclSobel")
// kernels
, clSobel(*this)
{
	addSourceFile("filter/oclSobel.cl");

	exportKernel(clSobel);
}

//
//
//

int oclSobel::compile()
{
	clSobel = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clSobel = createKernel("clSobel");
	KERNEL_VALIDATE(clSobel)
	return 1;
}

//
//
//

int oclSobel::compute(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDx, oclImage2D& bfDy)
{
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clSobel.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

    clSetKernelArg(clSobel, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clSobel, 1, sizeof(cl_mem), bfDx);
	clSetKernelArg(clSobel, 2, sizeof(cl_mem), bfDy);
 	clSetKernelArg(clSobel, 3, sizeof(cl_uint), &lIw);
	clSetKernelArg(clSobel, 4, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clSobel, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clSobel.getEvent());

	ENQUEUE_VALIDATE
	return true;
}
