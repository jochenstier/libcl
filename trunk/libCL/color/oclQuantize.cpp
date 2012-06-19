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

#include "oclQuantize.h"

oclQuantize::oclQuantize(oclContext& iContext)
: oclProgram(iContext, "oclQuantize")
// kernels
, clQuantizeLAB(*this)
{
    addSourceFile("color/oclQuantize.cl");

    exportKernel(clQuantizeLAB);
}

//
//
//

int oclQuantize::compile()
{
	clQuantizeLAB = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

    clQuantizeLAB = createKernel("clQuantizeLAB");
	KERNEL_VALIDATE(clQuantizeLAB)
	return 1;
}


int oclQuantize::quantizeLAB(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, float ibinL, float ibinA, float ibinB, float iSharpness)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfSrce.dim(0);
    lGlobalSize[1] = bfSrce.dim(1);
	clSetKernelArg(clQuantizeLAB, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clQuantizeLAB, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clQuantizeLAB, 2, sizeof(cl_float), &ibinL);
	clSetKernelArg(clQuantizeLAB, 3, sizeof(cl_float), &ibinA);
	clSetKernelArg(clQuantizeLAB, 4, sizeof(cl_float), &ibinB);
	clSetKernelArg(clQuantizeLAB, 5, sizeof(cl_float), &iSharpness);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clQuantizeLAB, 2, NULL, lGlobalSize, 0, 0, NULL, clQuantizeLAB.getEvent());
	ENQUEUE_VALIDATE
    return 1;
};
