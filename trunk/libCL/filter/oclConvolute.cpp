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
#define _USE_MATH_DEFINES
#include <math.h>

#include "oclConvolute.h"

static cl_float sGauss3[3] = { 1.0f/4.0f, 2.0f/4.0f, 1.0f/4.0f };
static cl_float sGauss5[5] = { 1.0f/17.0f, 4.0f/17.0f, 7.0f/17.0f, 4.0f/17.0f, 1.0f/17.0f };
static cl_float sdoG5[5];
static cl_float sdoG7[7];
static cl_float sdoG9[9];

void calcGauss1D(float iSigma, float* iBuffer, int iWidth)
{
    float t1 = 1.0/pow(sqrt(2.0f*M_PI)*iSigma, 1);
    float t2 = 1.0f/(2.0f*iSigma*iSigma);
    int r = iWidth/2;
    float total = 0.0f;
    for (int i=-r; i<=r; i++) 
    {
        float v = t1*exp(-i*i*t2);
        iBuffer[r+i] = v;
        total += v;
    }
    for (int i=-r; i<=r; i++) 
    {
        iBuffer[r+i]/=total;
    }
}

void calcGauss2D(float iSigma, float* iBuffer, int iWidth, int iHeigth)
{
    float t1 = 1.0f/pow(sqrt(2.0f*M_PI)*iSigma, 2);
    float t2 = 1.0f/(2.0f*iSigma*iSigma);
    float total = 0.0f;
    int rx = iWidth/2;
    int ry = iHeigth/2;
    for (int x=-rx; x<=rx; x++) 
    {
        for (int y=-ry; y<=ry; y++) 
        {
            float d = sqrt((float)(x*x+y*y));
            float v = t1*exp(-d*d*t2); 
            iBuffer[(ry+y)*iWidth+(rx+x)] = v;
            total += v;
       }
    }
    for (int x=-rx; x<=rx; x++) 
    {
        for (int y=-ry; y<=ry; y++) 
        {
            iBuffer[(ry+y)*iWidth+(rx+x)]/=total;
        }
    }
}

oclConvolute::oclConvolute(oclContext& iContext)
: oclProgram(iContext, "oclConvolute")
// kernels
, clIso2D(*this)
, clIso2Dsep(*this)
, clAniso2Dtang(*this)
, clAniso2Dorth(*this)
{
    exportKernel(clIso2D);
    exportKernel(clIso2Dsep);
    exportKernel(clAniso2Dtang);
    exportKernel(clAniso2Dorth);

    addSourceFile("filter\\oclConvolute.cl");
}

//
//
//

int oclConvolute::compile()
{
	clIso2D = 0;
	clIso2Dsep = 0;
	clAniso2Dtang = 0;
	clAniso2Dorth = 0;

	if (!oclProgram::compile())
	{
		return 0;
	}

	clIso2D = createKernel("clIso2D");
	KERNEL_VALIDATE(clIso2D)
	clIso2Dsep = createKernel("clIso2Dsep");
	KERNEL_VALIDATE(clIso2Dsep)

	clAniso2Dtang = createKernel("clAniso2Dtang");
	KERNEL_VALIDATE(clAniso2Dtang)
	clAniso2Dorth = createKernel("clAniso2Dorth");
	KERNEL_VALIDATE(clAniso2Dorth)
	return 1;
}

//
//
//

bool oclConvolute::gauss1D(float iSigma, oclBuffer& iBuffer)
{
    if (iBuffer.map(CL_MAP_WRITE))
    {
        int lKernelW = iBuffer.dim(0)/sizeof(cl_float);
        if (lKernelW < 3 || lKernelW % 2 == 0)
        {
            Log(ERR) << "Invalid buffer size of gauss1D size = " << iBuffer.dim(0)/sizeof(cl_float) << "*cl_float. Must be > 2 and odd";
            return false;
        }
        calcGauss1D(iSigma,  iBuffer.ptr<cl_float>(), lKernelW);
        iBuffer.unmap();
        return true;
    }
    return false;
}

bool oclConvolute::gauss2D(float iSigma, oclBuffer& iBuffer, int iKernelW, int iKernelH)
{
    if (iBuffer.map(CL_MAP_WRITE))
    {
        if (iKernelW % 2 == 0 || iKernelH % 2 == 0)
        {
            Log(ERR) << "Invalid buffer size of DoG2D." << iBuffer.dim(0)/sizeof(cl_float) << " Kernel dimensions must be odd";
            return false;
        }
        int lSize = iBuffer.dim(0)/sizeof(cl_float);
        if (iKernelW*iKernelH > lSize)
        {
            Log(ERR) << "Invalid buffer size for given kernel dimension :" << iBuffer.dim(0)/sizeof(cl_float) << "*cl_float";
            return false;
        }
        calcGauss2D(iSigma,  iBuffer.ptr<cl_float>(), iKernelW, iKernelH);
        iBuffer.unmap();
        return true;
    }
    return false;
}

//
//
//

bool oclConvolute::DoG1D(float iSigmaA, float iSigmaB, float iSensitivity, oclBuffer& iBuffer)
{
    if (iBuffer.map(CL_MAP_WRITE))
    {
        int lKernelW = iBuffer.dim(0)/sizeof(cl_float);
        if (lKernelW < 3 || lKernelW % 2 == 0)
        {
            Log(ERR) << "Invalid buffer size of DoG1D size = " << iBuffer.dim(0)/sizeof(cl_float) << "*cl_float. Must be > 2 and odd";
            return false;
        }
        
        cl_float* lGaussA = new cl_float[lKernelW];
        calcGauss1D(iSigmaA, lGaussA, lKernelW);
        cl_float* lGaussB = new cl_float[lKernelW];
        calcGauss1D(iSigmaB, lGaussB, lKernelW);
        cl_float* lBuffer = iBuffer.ptr<cl_float>();

        cl_float total = 0;
        for (int i=0; i<lKernelW; i++)
        {
            lBuffer[i] = (lGaussA[i] - iSensitivity*lGaussB[i]);
            total += lBuffer[i];
            //Log(WARN) << lBuffer[i];
        }
        total /= lKernelW;
        for (int i=0; i<lKernelW; i++)
        {
            lBuffer[i] -= total;
        }


        iBuffer.unmap();
        delete lGaussA;
        delete lGaussB;
        return true;
    }
    return false;
}

bool oclConvolute::DoG2D(float iSigmaA, float iSigmaB, float iSensitivity, oclBuffer& iBuffer, int iKernelW, int iKernelH)
{
    if (iBuffer.map(CL_MAP_WRITE))
    {
        if (iKernelW % 2 == 0 || iKernelH % 2 == 0)
        {
            Log(ERR) << "Invalid buffer size of DoG2D." << iBuffer.dim(0)/sizeof(cl_float) << " Kernel dimensions must be odd";
            return false;
        }
        int lSize = iBuffer.dim(0)/sizeof(cl_float);
        if (iKernelW*iKernelH > lSize)
        {
            Log(ERR) << "Invalid buffer size for given kernel dimension :" << iBuffer.dim(0)/sizeof(cl_float) << "*cl_float";
            return false;
        }
        
        cl_float* lGaussA = new cl_float[iKernelW*iKernelH];
        calcGauss2D(iSigmaA, lGaussA, iKernelW, iKernelH);
        cl_float* lGaussB = new cl_float[iKernelW*iKernelH];
        calcGauss2D(iSigmaB, lGaussB, iKernelW, iKernelH);
        cl_float* lBuffer = iBuffer.ptr<cl_float>();
        cl_float total = 0;
        for (int i=0; i<iKernelW*iKernelH; i++)
        {
            lBuffer[i] = (lGaussA[i] - iSensitivity*lGaussB[i]);
            total += lBuffer[i];
        }
        total /= iKernelW*iKernelH;
        for (int i=0; i<iKernelW*iKernelH; i++)
        {
            lBuffer[i] -= total;
        }

        iBuffer.unmap();
        delete lGaussA;
        delete lGaussB;
        return true;
    }
    return false;
}

//
//
//

int oclConvolute::iso2D(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclBuffer& bfFilter, int iFilterW, int iFilterH)
{
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clIso2D.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

	clSetKernelArg(clIso2D, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clIso2D, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clIso2D, 2, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clIso2D, 3, sizeof(cl_uint), &iFilterW);
	clSetKernelArg(clIso2D, 4, sizeof(cl_uint), &iFilterH);
	clSetKernelArg(clIso2D, 5, sizeof(cl_uint), &lIw);
	clSetKernelArg(clIso2D, 6, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clIso2D, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clIso2D.getEvent());
	ENQUEUE_VALIDATE
	return true;
}   

int oclConvolute::iso2Dsep(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int2 iAxis, oclBuffer& bfFilter)
{
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clIso2Dsep.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

    cl_int lFilterSize = bfFilter.dim(0)/sizeof(cl_float);
    if (lFilterSize %2 == 0)
    {
        Log(ERR, this) << "Failure in call to oclConvolute::iso2D : kernel size must be odd ";
    }
	clSetKernelArg(clIso2Dsep, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clIso2Dsep, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clIso2Dsep, 2, sizeof(cl_int2),  &iAxis);
	clSetKernelArg(clIso2Dsep, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clIso2Dsep, 4, sizeof(cl_int), &lFilterSize);
 	clSetKernelArg(clIso2Dsep, 5, sizeof(cl_uint), &lIw);
	clSetKernelArg(clIso2Dsep, 6, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clIso2Dsep, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clIso2Dsep.getEvent());
	ENQUEUE_VALIDATE
	return true;
}   



int oclConvolute::aniso2Dtang(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclImage2D& bfLine, oclBuffer& bfFilter)
{	
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clAniso2Dtang.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

    cl_int lFilterSize = bfFilter.dim(0)/sizeof(cl_float);
    if (lFilterSize %2 == 0)
    {
        Log(ERR, this) << "Failure in call to oclConvolute::iso2D : kernel size must be odd ";
    }

	clSetKernelArg(clAniso2Dtang, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clAniso2Dtang, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clAniso2Dtang, 2, sizeof(cl_mem), bfLine);
	clSetKernelArg(clAniso2Dtang, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clAniso2Dtang, 4, sizeof(cl_uint), &lFilterSize);
 	clSetKernelArg(clAniso2Dtang, 5, sizeof(cl_uint), &lIw);
	clSetKernelArg(clAniso2Dtang, 6, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clAniso2Dtang, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clAniso2Dtang.getEvent());
	ENQUEUE_VALIDATE

	return true;
}   

int oclConvolute::aniso2Dorth(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclImage2D& bfLine, oclBuffer& bfFilter)
{	
	cl_uint lIw = bfSrce.getImageInfo<size_t>(CL_IMAGE_WIDTH);
	cl_uint lIh = bfSrce.getImageInfo<size_t>(CL_IMAGE_HEIGHT);
	size_t lGlobalSize[2];
	size_t lLocalSize[2];
    clAniso2Dorth.localSize2D(iDevice, lGlobalSize, lLocalSize, lIw, lIh);

    cl_int lFilterSize = bfFilter.dim(0)/sizeof(cl_float);
    if (lFilterSize %2 == 0)
    {
        Log(ERR, this) << "Failure in call to oclConvolute::iso2D : kernel size must be odd ";
    }

	clSetKernelArg(clAniso2Dorth, 0, sizeof(cl_mem), bfSrce);
	clSetKernelArg(clAniso2Dorth, 1, sizeof(cl_mem), bfDest);
	clSetKernelArg(clAniso2Dorth, 2, sizeof(cl_mem), bfLine);
	clSetKernelArg(clAniso2Dorth, 3, sizeof(cl_mem), bfFilter);
	clSetKernelArg(clAniso2Dorth, 4, sizeof(cl_uint), &lFilterSize);
 	clSetKernelArg(clAniso2Dorth, 5, sizeof(cl_uint), &lIw);
	clSetKernelArg(clAniso2Dorth, 6, sizeof(cl_uint), &lIh);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clAniso2Dorth, 2, NULL, lGlobalSize, lLocalSize, 0, NULL, clAniso2Dorth.getEvent());
	ENQUEUE_VALIDATE

    return true;
}   
