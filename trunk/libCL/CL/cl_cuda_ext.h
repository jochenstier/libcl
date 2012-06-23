/**********************************************************************************
 * Copyright 1993-2009 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 **********************************************************************************/

#ifndef __CL_CUDA_EXT_H
#define __CL_CUDA_EXT_H

#include <cuda.h>
#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * cl_nv_cuda_interop_3_1                                                     */
/******************************************************************************/

#define CL_COMMAND_CUDA_OPERATIONS                  0x402D


typedef CL_API_ENTRY cl_int (CL_API_CALL *clAcquireCUDAContextNV_fn)(
    cl_context             ctx,
    CUcontext              *cuCtx,
    cl_device_id           deviceID,
    cl_uint                num_event_to_wait_for,
    cl_event               *eventsToWait) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clReleaseCUDAContextNV_fn)(
    cl_context            ctx,
    CUcontext             cuCtx,
    cl_event              *event) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clAcquireCUDAPointerFromBufferNV_fn)(
    CUcontext       cuCtx,
    cl_mem          mem,
    void            **devicePtr ) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clReleaseCUDAPointerFromBufferNV_fn)(
    cl_mem          mem ) CL_API_SUFFIX__VERSION_1_0;


#ifdef __cplusplus
}
#endif

#endif  // __CL_CUDA_EXT_H

