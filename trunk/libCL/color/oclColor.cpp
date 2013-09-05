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

#include "oclColor.h"

oclColor::oclColor(oclContext& iContext, oclProgram* iParent)
: oclProgram(iContext, "oclColor", iParent)
// kernels
, clHSVtoRGB(*this, "clHSVtoRGB")
, clRGBtoHSV(*this, "clRGBtoHSV")

, clRGBtoXYZ(*this, "clRGBtoXYZ")
, clXYZtoRGB(*this, "clXYZtoRGB")

, clRGBtoLAB(*this, "clRGBtoLAB")
, clLABtoRGB(*this, "clLABtoRGB")
{
    addSourceFile("color/oclColor.cl");
}


int oclColor::invoke(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclKernel& iKernel)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfSrce.dim(0);
    lGlobalSize[1] = bfSrce.dim(1);
    clSetKernelArg(iKernel, 0, sizeof(cl_mem), bfSrce);
    clSetKernelArg(iKernel, 1, sizeof(cl_mem), bfDest);
    sStatusCL = clEnqueueNDRangeKernel(iDevice, iKernel, 2, NULL, lGlobalSize, 0, 0, NULL, iKernel.getEvent());
    ENQUEUE_VALIDATE
    return 1;
};
