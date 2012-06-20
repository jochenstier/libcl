// TEST.cpp : Defines the entry point for the console application.
//
#include "console.h"

#include <math.h>
#include <time.h>

#ifdef WIN32
#include "stdafx.h"
#include <windows.h>

#else

#include <time.h>
typedef unsigned long DWORD;

DWORD GetTickCount() {
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts) != 0) {
        //error
        return -1;
    }
    return ts.tv_sec*1000;
}

#endif

#include "oclContext.h"

#include "sort/oclRadixSort.h"
#include "phys/oclFluid3D.h"
#include "geom/oclBvhTrimesh.h"

#include "filter/oclRecursiveGaussian.h"
#include "filter/oclBilateral.h"
#include "filter/oclSobel.h"
#include "filter/oclTangent.h"
#include "filter/oclBilinearPyramid.h"
#include "filter/oclConvolute.h"

#include "color/oclColor.h"
#include "color/oclQuantize.h"

#include "image/oclToneMapping.h"
#include "image/oclBloom.h"
#include "image/oclAmbientOcclusion.h"


void testRadixSort(oclContext& iContext);
void testFluid3D0(oclContext& iContext);
void testFluid3D1(oclContext& iContext);
void testBvhTrimesh(oclContext& iContext);
void testCompile(oclContext& iContext);


//
// Main
//
#ifdef WIN32
int _tmain(int argc, _TCHAR* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    // libCL requires a root path to find the .cl files	
    oclInit("../"); 

    // find the first available platform
    oclContext* lContext = oclContext::create(
                                              oclContext::VENDOR_NVIDIA, 
                                              CL_DEVICE_TYPE_GPU);
    if (!lContext)
    {
        lContext = oclContext::create(
                                      oclContext::VENDOR_AMD, 
                                      CL_DEVICE_TYPE_GPU);
        if (!lContext)
        {
            lContext = oclContext::create(
                                          oclContext::VENDOR_INTEL, 
                                          CL_DEVICE_TYPE_CPU);
            if (!lContext)
            {
                lContext = oclContext::create(
                                              oclContext::VENDOR_AMD, 
                                              CL_DEVICE_TYPE_CPU);
            }
        }
    }


    if (!lContext)
    {
        Log(ERR) << "no OpenCL capable platform detected";
        return -1; 
    }
    else 
    {
        Log(INFO) << "\n*\n*\n*" <<
            "Running LibCL 1.0 using vendor platform: " << 
            lContext->getName() << "\n*\n*";
    }
    Log(INFO) << "\n\n";

    // run tests
    Log(INFO) << "****** calling radix sort ...";
    testRadixSort(*lContext);
    Log(INFO) << "****** done\n";

    Log(INFO) << "****** calling fluid test 0 ...";
    testFluid3D0(*lContext);
    Log(INFO) << "****** done\n";

    Log(INFO) << "****** calling fluid test 1 ...";
    testFluid3D1(*lContext);
    Log(INFO) << "****** done\n";

    Log(INFO) << "****** calling BVH construction ...";
    testBvhTrimesh(*lContext);
    Log(INFO) << "****** done\n";

    Log(INFO) << "****** compiling all ...";
    testCompile(*lContext);
    Log(INFO) << "****** done\n";

    return 0;
}

//
// Radix Sort Test
//

void testRadixSort(oclContext& iContext)
{
    oclDevice& lDevice = iContext.getDevice(0);
    oclBuffer bfKey(iContext, "bfKey");
    oclBuffer bfVal(iContext, "bfVal");

    bfKey.create<cl_uint> (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1024);
    bfVal.create<cl_uint> (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1024);

    if (!bfKey.map(CL_MAP_READ | CL_MAP_WRITE))
    {
        return;
    }
    if (!bfVal.map(CL_MAP_READ | CL_MAP_WRITE))
    {
        return;
    }

    cl_uint* ptrKey = bfKey.ptr<cl_uint>();
    cl_uint* ptrVal = bfVal.ptr<cl_uint>();
    for (int i=0; i<1024; i++)
    {
        ptrKey[i] = rand();
        ptrVal[i] = i;
    }
    bfVal.write();
    bfKey.write();

    oclRadixSort clProgram(iContext);
    if (clProgram.compile())
    {
        clProgram.compute(lDevice, bfKey, bfVal, 0, 32);

        bfVal.read();
        bfKey.read();
        for (int i=1; i<1024; i++)
        {
            if (ptrKey[i] < ptrKey[i-1])
            {
                Log(WARN) << "array not sorted";
                break;
            }
        }
    }

    bfVal.unmap();
    bfKey.unmap();
    Log(INFO) << "testRadixSort completed";
}


//
// Fluid3D Test
//

void testFluid3D0(oclContext& iContext)
{
    oclDevice& lDevice = iContext.getDevice(0);

    oclFluid3D clProgram(iContext);
    if (clProgram.compile())
    {
        oclBuffer* lBuffer = clProgram.getPositionBuffer();
        unsigned int lCount = clProgram.getParticleCount();

        if (lBuffer->map(CL_MAP_WRITE))
        {
            // initialize particle positions
            cl_float4* lPtr = lBuffer->ptr<cl_float4>();
            for (unsigned int i=0; i<lCount; i++)
            {
                lPtr[i].s[0] = (float)rand()/RAND_MAX-0.5;
                lPtr[i].s[1] = (float)rand()/RAND_MAX-0.5;
                lPtr[i].s[2] = (float)rand()/RAND_MAX-0.5;
                lPtr[i].s[3] = 0;
            }
            lBuffer->unmap();
        }

        Log(INFO) << "Computing particles for 7 seconds ";
        DWORD lEnd = GetTickCount() + 7000;
        while (lEnd > GetTickCount())
        {
            clProgram.compute(lDevice);
            clFinish(lDevice);
            cout << ".";
        }
        cout << "\n";

        if (lBuffer->map(CL_MAP_READ))
        {
            // compute average paricle position
            cl_float4 lAvg = { 0,0,0,0};
            cl_float4* lPtr = lBuffer->ptr<cl_float4>();
            for (unsigned int i=0; i<lCount; i++)
            {
                lAvg.s[0] += lPtr[i].s[0];
                lAvg.s[1] += lPtr[i].s[1];
                lAvg.s[2] += lPtr[i].s[2];
            }
            lAvg.s[0] /= lCount;
            lAvg.s[1] /= lCount;
            lAvg.s[2] /= lCount;
            Log(INFO) << "Average particle position = " << *lPtr;
            lBuffer->unmap();
        }
    }
}

//
// Fluid3D with callback
//

void testFluid3D1(oclContext& iContext)
{
    oclDevice& lDevice = iContext.getDevice(0);

    oclFluid3D clProgram(iContext);
    if (clProgram.compile())
    {
        oclBuffer* lBuffer = clProgram.getPositionBuffer();
        unsigned int lCount = clProgram.getParticleCount();

        if (lBuffer->map(CL_MAP_WRITE))
        {
            // initialize particle positions
            cl_float4* lPtr = lBuffer->ptr<cl_float4>();
            for (unsigned int i=0; i<lCount; i++)
            {
                lPtr[i].s[0] = (float)rand()/RAND_MAX-0.5;
                lPtr[i].s[1] = (float)rand()/RAND_MAX-0.5;
                lPtr[i].s[2] = (float)rand()/RAND_MAX-0.5;
                lPtr[i].s[3] = 0;
            }
            lBuffer->unmap();
        }

        // implement event handler structure
        struct srtFluid : public srtEvent
        {
            srtFluid(oclFluid3D& iFluid, oclDevice& iDevice)
            : srtEvent(oclFluid3D::EVT_INTEGRATE)
            , clGravity(iFluid.getKernel("clGravity"))
            , clIntegrateVelocity(iFluid.getKernel("clIntegrateVelocity"))
            , clIntegrateForce(iFluid.getKernel("clIntegrateForce"))
            , mDevice(iDevice)
            , mFluid(iFluid)
            {
            }
            bool operator() (oclProgram& iSource) 
            { 
                // call all the fluid kernels except clipBox
                size_t lPartcleCount = mFluid.getParticleCount();
		        sStatusCL = clEnqueueNDRangeKernel(mDevice, *clIntegrateForce, 1, NULL, &lPartcleCount, &mFluid.cLocalSize, 0, NULL, clIntegrateForce->getEvent());
		        if (!oclSuccess("clEnqueueNDRangeKernel", &mFluid))
                {
                    return false;
                }
		        sStatusCL = clEnqueueNDRangeKernel(mDevice, *clGravity, 1, NULL, &lPartcleCount, &mFluid.cLocalSize, 0, NULL, clGravity->getEvent());
		        if (!oclSuccess("clEnqueueNDRangeKernel", &mFluid))
                {
                    return false;
                }
		        sStatusCL = clEnqueueNDRangeKernel(mDevice, *clIntegrateVelocity, 1, NULL, &lPartcleCount, &mFluid.cLocalSize, 0, NULL, clIntegrateVelocity->getEvent());
		        if (!oclSuccess("clEnqueueNDRangeKernel", &mFluid))
                {
                    return false;
                }
                return 1;
            }

            oclDevice& mDevice;
            oclFluid3D& mFluid;

            oclKernel* clGravity;
            oclKernel* clIntegrateVelocity;
            oclKernel* clIntegrateForce;

        } evtHandler(clProgram, lDevice);

        clProgram.addEventHandler(evtHandler);

        Log(INFO) << "Computing particles for 7 seconds ";
        DWORD lEnd = GetTickCount() + 7000;
        while (lEnd > GetTickCount())
        {
            clProgram.compute(lDevice);
            clFinish(lDevice);
            cout << ".";
        }
        cout << "\n";

        if (lBuffer->map(CL_MAP_READ))
        {
            // compute average paricle position
            cl_float4 lAvg = { 0,0,0,0};
            cl_float4* lPtr = lBuffer->ptr<cl_float4>();
            for (unsigned int i=0; i<lCount; i++)
            {
                lAvg.s[0] += lPtr[i].s[0];
                lAvg.s[1] += lPtr[i].s[1];
                lAvg.s[2] += lPtr[i].s[2];
            }
            lAvg.s[0] /= lCount;
            lAvg.s[1] /= lCount;
            lAvg.s[2] /= lCount;
            Log(INFO) << "Average particle position = " << *lPtr;
            lBuffer->unmap();
        }
    }
}

//
// Test BvhTrimesh
//

void testBvhTrimesh(oclContext& iContext)
{
    oclDevice& lDevice = iContext.getDevice(0);

    oclBuffer bfVertex(iContext, "bfVertex");
    bfVertex.create<cl_float4> (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1000);
    if (bfVertex.map(CL_MAP_WRITE))
    {
        // initiaize vertices
        cl_float4* lPtr = bfVertex.ptr<cl_float4>();
        for (unsigned int i=0; i<1000; i++)
        {
            lPtr[i].s[0] = (float)rand()/RAND_MAX-0.5;
            lPtr[i].s[1] = (float)rand()/RAND_MAX-0.5;
            lPtr[i].s[2] = (float)rand()/RAND_MAX-0.5;
            lPtr[i].s[3] = 1;
        }
        bfVertex.unmap();
    }
    else return;


    oclBuffer bfIndex(iContext, "bfIndex");
    bfIndex.create<cl_uint> (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1000);
    if (bfIndex.map(CL_MAP_WRITE))
    {
        // initiaize indecies
        cl_uint* lPtr = bfIndex.ptr<cl_uint>();
        for (unsigned int i=0; i<1000; i++)
        {
            lPtr[i] = i;
        }
        bfIndex.unmap();
    }
    else return;

    oclBvhTrimesh clProgram(iContext);
    
    if (clProgram.compile())
    {
        if (clProgram.compute(lDevice, bfVertex, bfIndex))
        {
            cl_uint lRootNode = clProgram.getRootNode();

            oclBuffer& lNodes = clProgram.getNodeBuffer();
            if (lNodes.map(CL_MAP_READ))
            {
                oclBvhTrimesh::BVHNode* lPtr = lNodes.ptr<oclBvhTrimesh::BVHNode>();
                Log(INFO) << "BVH Root (min):" << lPtr[lRootNode].bbMin;
                Log(INFO) << "BVH Root (max):" << lPtr[lRootNode].bbMax;
                Log(INFO) << "BVH Root (left):" << lPtr[lRootNode].left;
                Log(INFO) << "BVH Root (right):" << lPtr[lRootNode].right;
                lNodes.unmap();
            }
        } else {
            Log(ERR) << "Something Failed, if running on AMD it's OK\n";
        }
    }
};




void testCompile(oclContext& iContext)
{
    oclDevice& lDevice = iContext.getDevice(0);

    // fitler
    oclConvolute clConvolute(iContext);
    clConvolute.compile();
    oclBilateral clBilateral(iContext);
    clBilateral.compile();
    oclBilinearPyramid clBilinearPyramid(iContext);
    clBilinearPyramid.compile();
    oclRecursiveGaussian clRecursiveGaussian(iContext);
    clRecursiveGaussian.compile();
    oclSobel clSobel(iContext);
    clSobel.compile();
    oclTangent clTangent(iContext);
    clTangent.compile();

    // color
    oclColor clColor(iContext);
    clColor.compile();
    oclQuantize clQuantize(iContext);
    clQuantize.compile();

    // image
    oclToneMapping clToneMapping(iContext);
    clToneMapping.compile();
    oclBloom clBloom(iContext);
    clBloom.compile();
    oclAmbientOcclusion clAmbientOcclusion(iContext);
    clAmbientOcclusion.compile();
}
