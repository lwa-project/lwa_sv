/*
 * Copyright (c) 2021, The Bifrost Authors. All rights reserved.
 # Copyright (c) 2021, The University of New Mexico. All rights reserved.

 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#include <bifrost/array.h>
#include <bifrost/common.h>
#include "utils.hpp"
#include "cuda.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda_fp16.h>

#include "btcc.h"

#include "libtcc/Correlator.h"

thread_local cudaStream_t g_cuda_stream = cudaStreamPerThread;

//
// Convert from a TCC bit depth to a Bifrost data type
//
inline BFdtype bf_dtype_from_tcc(int nr_bits) {
    switch(nr_bits) {
        case 4: return BF_DTYPE_CI4;
        case 8: return BF_DTYPE_CI8;
        case 16: return BF_DTYPE_CF16;
        default: return BF_DTYPE_CF32;
    }
}

template<typename DType>
__global__ void swizzel_kernel(int          ntime,
                               int          nchan,
                               int          nstand,
                               int          npol,
                               int          ntime_per_block,
                               const DType* in,
                               DType*       out) {
    int t, f, s, p;
    t = blockIdx.x;
    f = blockIdx.y;
    s = threadIdx.x;
    
    int t0, t1;
    t0 = t / ntime_per_block;
    t1 = t % ntime_per_block;
    
    int in_idx = t*nchan*nstand*npol + f*nstand*npol + s*npol;
    int out_idx = f*ntime*nstand*npol + t0*nstand*npol*ntime_per_block \
                  + s*npol*ntime_per_block;
    #pragma unroll
    for(p=0; p<npol; p++) {
        out[out_idx + p*ntime_per_block + t1] = in[in_idx + p];
    }
}

template<typename DType>
inline void launch_swizzel_kernel(int          ntime,
                                  int          nchan, 
                                  int          nstand, 
                                  int          npol,
                                  int          ntime_per_block,
                                  DType*       d_in,
                                  DType*       d_out,
                                  cudaStream_t stream=0) {
    dim3 block(nstand, 1);
    dim3 grid(ntime, nchan, 1);
    void* args[] = {&ntime,
                    &nchan,
                    &nstand,
                    &npol,
                    &ntime_per_block,
                    &d_in,
                    &d_out};
    BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)swizzel_kernel<DType>,
                                             grid, block,
                                             &args[0], 0, stream),
                            BF_STATUS_INTERNAL_ERROR);
}

template<typename DType>
__global__ void accumulate_kernel(int          nchan,
                                  int          nstand,
                                  int          npolprod,
                                  const DType* in,
                                  DType*       out) {
    int f, s, p, c;
    f = blockIdx.x;
    s = threadIdx.x;
    
    int b = f*(nstand+1)*(nstand/2) + s*(2*(nstand-1)+1-s)/2 + s;
    for(int i=b; i<b+nstand-s; i++) {
        #pragma unroll
        for(p=0; p<npolprod; p++) {
            #pragma unroll
            for(c=0; c<2; c++) {
                out[i*npolprod*2 + p*2 + c] += in[i*npolprod*2 + p*2 + c];
            }
        }
    }
}

template<typename DType>
inline void launch_accumulate_kernel(int          nchan, 
                                     int          nstand, 
                                     int          npolprod,
                                     DType*       d_in,
                                     DType*       d_out,
                                     cudaStream_t stream=0) {
    dim3 block(nstand, 1);
    dim3 grid(nchan, 1, 1);
    void* args[] = {&nchan,
                    &nstand,
                    &npolprod,
                    &d_in,
                    &d_out};
    BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)accumulate_kernel<DType>,
                                             grid, block,
                                             &args[0], 0, stream),
                            BF_STATUS_INTERNAL_ERROR);
}

template<typename DType>
__global__ void reorder_kernel(int          nchan,
                               int          nstand,
                               int          npol,
                               const DType* in,
                               DType*       out) {
    int f, i, j, pol1, pol2;
    f = blockIdx.x;
    j = threadIdx.x;

    for (i=0; i<=j; i++) {
        int k = f*(nstand+1)*(nstand/2) + j*(j+1)/2 + i;
        int ku = f*(nstand+1)*(nstand/2) + i*(2*(nstand-1)+1-i)/2 + j;
        #pragma unroll
        for (pol1=0; pol1<npol; pol1++) {
            #pragma unroll
            for (pol2=0; pol2<npol; pol2++) {
                size_t index = ((k*npol+pol1)*npol+pol2)*2;
                size_t indexu = ((ku*npol+pol2)*npol+pol1)*2;
                out[indexu + 0] =  in[index + 0];
                out[indexu + 1] = -in[index + 1];
            }
        }
    }
}

template<typename DType>
inline void launch_reorder_kernel(int          nchan, 
                                  int          nstand, 
                                  int          npol,
                                  DType*       d_in,
                                  DType*       d_out,
                                  cudaStream_t stream=0) {
    dim3 block(nstand, 1);
    dim3 grid(nchan, 1, 1);
    void* args[] = {&nchan,
                    &nstand,
                    &npol,
                    &d_in,
                    &d_out};
    BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)reorder_kernel<DType>,
                                             grid, block,
                                             &args[0], 0, stream),
                            BF_STATUS_INTERNAL_ERROR);
}

class btcc_impl {
private:
    int _nbits;
    int _ntime;
    int _nchan;
    int _nstand;
    int _npol;
    int _ntime_per_block;
    
    tcc::Correlator* _tcc;
    void* _reordered = NULL;
    void* _accum = NULL;
    cudaStream_t _stream;
    
public:
    btcc_impl() : _tcc(NULL), _reordered(NULL), _accum(NULL), _stream(g_cuda_stream) {}
    ~btcc_impl() {
        cudaDeviceSynchronize();
        
        if(_tcc) {
           delete _tcc;
        }
        if(_reordered) {
            cudaFree(_reordered);
        }
        if(_accum) {
            cudaFree(_accum);
        }
    }
    inline int ntime() const { return _ntime; }
    inline int nchan() const { return _nchan; }
    inline int nstand() const { return _nstand; }
    inline int npol() const { return _npol; }
    inline int ntime_per_block() const { return _ntime_per_block; }
    inline int nbaseline() const { return (_nstand+1)*(_nstand/2); }
    inline BFdtype in_dtype() const { return bf_dtype_from_tcc(_nbits); }
    inline BFdtype out_dtype() const { return _nbits == 16 ? BF_DTYPE_CF32 : BF_DTYPE_CI32; }
    void init(int nbits, int ntime, int nchan, int nstand, int npol) {
        _nbits = nbits;
        _ntime = ntime;
        _nchan = nchan;
        _nstand = nstand;
        _npol = npol;
        _ntime_per_block = 128 / _nbits;
        
        // Sanity checks
        BF_ASSERT_EXCEPTION((_nbits == 4) || (_nbits == 8) || (_nbits == 16), BF_STATUS_UNSUPPORTED_DTYPE);
        BF_ASSERT_EXCEPTION(_ntime % _ntime_per_block == 0, BF_STATUS_UNSUPPORTED_SHAPE);
        
        // Setup the tensor core correlator
        _tcc = new tcc::Correlator(_nbits, _nstand, _nchan, _ntime, _npol);
        
        // Temporary storage for reordered input data and accumulation
        cudaMalloc(&_reordered, _ntime*_nchan*_nstand*_npol*_nbits*2);
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_MEM_ALLOC_FAILED);
        cudaMalloc(&_accum, _nchan*(_nstand+1)*(_nstand/2)*_npol*_npol*2*sizeof(float));
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_MEM_ALLOC_FAILED);
        
        // Zero out the accumulator
        this->reset_state();
    }
    void set_stream(cudaStream_t stream) {
        cudaDeviceSynchronize();
        
        _stream = stream;
    }
    void reset_state() {
        BF_ASSERT_EXCEPTION(_tcc, BF_STATUS_INVALID_STATE); 
        
        cudaMemset(_accum, 0, _nchan*_nstand*(_nstand+1)/2*_npol*_npol*2*sizeof(float));
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_MEM_OP_FAILED);
    }
    void exec(BFarray const* in, BFarray* out, BFbool dump) {
        BF_ASSERT_EXCEPTION(_tcc, BF_STATUS_INVALID_STATE); 
        
#define LAUNCH_SWIZZEL_KERNEL(DType) \
        launch_swizzel_kernel(_ntime, _nchan, _nstand, _npol, _ntime_per_block, \
                              (DType)in->data, (DType)_reordered, _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:  LAUNCH_SWIZZEL_KERNEL(signed char*); break;
            case BF_DTYPE_CI8:  LAUNCH_SWIZZEL_KERNEL(signed short*); break;
            case BF_DTYPE_CF16: LAUNCH_SWIZZEL_KERNEL(__half*); break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
        
#undef LAUNCH_SWIZZEL_KERNEL
        
        (*_tcc).launchAsync((CUstream) _stream, (CUdeviceptr) out->data, (CUdeviceptr) _reordered);
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        
#define LAUNCH_ACCUMULATE_KERNEL(DType) \
        launch_accumulate_kernel(_nchan, _nstand, _npol*_npol, \
                                 (DType)out->data, (DType)_accum, _stream)
                              
        switch( out->dtype ) {
            case BF_DTYPE_CI32: LAUNCH_ACCUMULATE_KERNEL(int*); break;
            case BF_DTYPE_CF32: LAUNCH_ACCUMULATE_KERNEL(float*); break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
          
#undef LAUNCH_ACCUMULATE_KERNEL
        
        if(dump) {
          
#define LAUNCH_REORDER_KERNEL(DType) \
          launch_reorder_kernel(_nchan, _nstand, _npol, \
                                (DType)_accum, (DType)out->data, _stream)
          
          switch( out->dtype ) {
              case BF_DTYPE_CI32: LAUNCH_REORDER_KERNEL(int*); break;
              case BF_DTYPE_CF32: LAUNCH_REORDER_KERNEL(float*); break;
              default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
          }
            
  #undef LAUNCH_REORDER_KERNEL
          
          this->reset_state();
        }
    }
};

BFstatus BTccCreate(btcc* plan_ptr) {
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new btcc_impl(),
                       *plan_ptr = 0);
}

BFstatus BTccInit(btcc  plan,
                  int   nbits,
                  int   ntime,
                  int   nchan,
                  int   nstand,
                  int   npol) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_TRY_RETURN(plan->init(nbits, ntime, nchan, nstand, npol));
}

BFstatus BTccSetStream(btcc        plan,
                       void const* stream) {
        BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
        BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
        BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}

BFstatus BTccResetState(btcc        plan) {
        BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
        BF_TRY_RETURN(plan->reset_state());
}

BFstatus BTccExecute(btcc           plan,
                     BFarray const* in,
                     BFarray*       out,
                     BFbool         dump) {
    BF_ASSERT(plan, BF_STATUS_INVALID_POINTER);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
  	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    
    BF_ASSERT( in->ndim == 3, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[0] == plan->ntime(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[1] == plan->nchan(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[2] == plan->nstand()*plan->npol(), BF_STATUS_INVALID_SHAPE);
    
    BF_ASSERT(out->ndim == 2, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[0] == plan->nchan(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[1] == plan->nbaseline()*plan->npol()*plan->npol(), BF_STATUS_INVALID_SHAPE);
    
    BF_ASSERT(in->dtype == plan->in_dtype(), BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(out->dtype == plan->out_dtype(), BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from(in->space, BF_SPACE_CUDA),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA),
              BF_STATUS_UNSUPPORTED_SPACE);
    
    BF_TRY_RETURN(plan->exec(in, out, dump));
}

BFstatus BTccDestroy(btcc plan) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
