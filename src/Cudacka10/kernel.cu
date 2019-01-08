// Cuckaroo Cycle, a memory-hard proof-of-work by John Tromp and team Grin
// Copyright (c) 2018 Jiri Photon Vadura and John Tromp
// This GGM miner file is covered by the FAIR MINING license

//Includes for IntelliSense
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <stdint.h>
#include <builtin_types.h>
#include <vector_functions.h>


typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u32 node_t;
typedef u64 nonce_t;


#define DUCK_SIZE_A 129LL
#define DUCK_SIZE_B 82LL

#define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL)

#define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL)

#define EDGE_BLOCK_SIZE (64)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)

#define EDGEBITS 29
#define NEDGES ((node_t)1 << EDGEBITS)
#define EDGEMASK (NEDGES - 1)

#define CTHREADS 512
#define BKTMASK4K (4096-1)
#define BKTGRAN 32

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  { \
    v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
    v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
    v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
    v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
  }

__device__ __forceinline__  void Increase2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	u32 old = atomicOr(ecounters + word, mask) & mask;

	if (old > 0)
		atomicOr(ecounters + word + 4096, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	return (ecounters[word + 4096] & mask) > 0;
}

extern "C" {
	__constant__ u64 recovery[42];

	__global__  void FluffySeed2A(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, ulonglong4 * buffer, int * indexes)
	{
		const int gid = blockDim.x * blockIdx.x + threadIdx.x;
		const int lid = threadIdx.x;

		__shared__ u64 tmp[64][16];
		__shared__ u32 counters[64];
		u64 sipblock[64];

		uint64_t v0, v1, v2, v3;

		if (lid < 64)
			counters[lid] = 0;

		__syncthreads();

		for (int i = 0; i < 1024*2; i += EDGE_BLOCK_SIZE)
		{
			u64 blockNonce = gid * (1024*2) + i;

			v0 = v0i;
			v1 = v1i;
			v2 = v2i;
			v3 = v3i;

			for (u32 b = 0; b < EDGE_BLOCK_SIZE; b++)
			{
				v3 ^= blockNonce + b;
				for (int r = 0; r < 2; r++)
					SIPROUND;
				v0 ^= blockNonce + b;
				v2 ^= 0xff;
				for (int r = 0; r < 4; r++)
					SIPROUND;

				sipblock[b] = (v0 ^ v1) ^ (v2  ^ v3);

			}
			u64 last = sipblock[EDGE_BLOCK_MASK];

			for (short s = 0; s < EDGE_BLOCK_SIZE; s++)
			{
				u64 lookup = s == EDGE_BLOCK_MASK ? last : sipblock[s] ^ last;
				uint2 hash = make_uint2(lookup & EDGEMASK, (lookup >> 32) & EDGEMASK);
				int bucket = hash.x & 63;
				
				__syncthreads();

				int counter = atomicAdd(counters + bucket, (u32)1);
				int counterLocal = counter % 16;
				tmp[bucket][counterLocal] = hash.x | ((u64)hash.y << 32);

				__syncthreads();

				if ( (counter > 0) && (counterLocal == 0 || counterLocal == 8))
				{
					int cnt = min((int)atomicAdd(indexes + bucket, 8), (int)(DUCK_A_EDGES_64 - 8));
					int idx = (bucket * DUCK_A_EDGES_64 + cnt) / 4;

					buffer[idx] = make_ulonglong4(
						atomicExch( &tmp[bucket][8 - counterLocal], 0),
						atomicExch( &tmp[bucket][9 - counterLocal], 0),
						atomicExch( &tmp[bucket][10 - counterLocal], 0),
						atomicExch( &tmp[bucket][11 - counterLocal], 0)
					);
					buffer[idx + 1] = make_ulonglong4(
						atomicExch( &tmp[bucket][12 - counterLocal], 0),
						atomicExch( &tmp[bucket][13 - counterLocal], 0),
						atomicExch( &tmp[bucket][14 - counterLocal], 0),
						atomicExch( &tmp[bucket][15 - counterLocal], 0)
					);
				}

			}
		}

		__syncthreads();

		if (lid < 64)
		{
			int counter = counters[lid];
			int counterBase = (counter % 16) >= 8 ? 8 : 0;
			int cnt = min((int)atomicAdd(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));
			int idx = (lid * DUCK_A_EDGES_64 + cnt) / 4;
			buffer[idx] = make_ulonglong4(tmp[lid][counterBase], tmp[lid][counterBase+1], tmp[lid][counterBase+2], tmp[lid][counterBase+3]);
			buffer[idx + 1] = make_ulonglong4(tmp[lid][counterBase+4], tmp[lid][counterBase+5], tmp[lid][counterBase+6], tmp[lid][counterBase+7]);
		}

	}
	__global__  void FluffySeed2B(const  uint2 * source, ulonglong4 * destination, const  int * sourceIndexes, int * destinationIndexes, int startBlock)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u64 tmp[64][16];
		__shared__ int counters[64];

		if (lid < 64)
			counters[lid] = 0;

		__syncthreads();

		const int offsetMem = startBlock * DUCK_A_EDGES_64;
		const int myBucket = group / BKTGRAN;
		const int microBlockNo = group % BKTGRAN;
		const int bucketEdges = min(sourceIndexes[myBucket + startBlock], (int)(DUCK_A_EDGES_64));
		const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN);
		const int loops = (microBlockEdgesCount / 128);

		for (int i = 0; i < loops; i++)
		{
			int edgeIndex = (microBlockNo * microBlockEdgesCount) + (128 * i) + lid;

			{
				uint2 edge = source[offsetMem + (myBucket * DUCK_A_EDGES_64) + edgeIndex];
				bool skip = (edgeIndex >= bucketEdges) || (edge.x == 0 && edge.y == 0);

				int bucket = (edge.x >> 6) & (64 - 1);

				__syncthreads();

				int counter = 0;
				int counterLocal = 0;

				if (!skip)
				{
					counter = atomicAdd(counters + bucket, (u32)1);
					counterLocal = counter % 16;
					tmp[bucket][counterLocal] = edge.x | ((u64)edge.y << 32);
				}

				__syncthreads();

				if ((counter > 0) && (counterLocal == 0 || counterLocal == 8))
				{
					int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + bucket, 8), (int)(DUCK_A_EDGES - 8));
					int idx = ((myBucket * 64 + bucket) * DUCK_A_EDGES + cnt) / 4;

					destination[idx] = make_ulonglong4(
						atomicExch(&tmp[bucket][8 - counterLocal], 0),
						atomicExch(&tmp[bucket][9 - counterLocal], 0),
						atomicExch(&tmp[bucket][10 - counterLocal], 0),
						atomicExch(&tmp[bucket][11 - counterLocal], 0)
					);
					destination[idx + 1] = make_ulonglong4(
						atomicExch(&tmp[bucket][12 - counterLocal], 0),
						atomicExch(&tmp[bucket][13 - counterLocal], 0),
						atomicExch(&tmp[bucket][14 - counterLocal], 0),
						atomicExch(&tmp[bucket][15 - counterLocal], 0)
					);
				}
			}
		}

		__syncthreads();

		if (lid < 64)
		{
			int counter = counters[lid];
			int counterBase = (counter % 16) >= 8 ? 8 : 0;
			int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), (int)(DUCK_A_EDGES - 8));
			int idx = ((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4;
			destination[idx] = make_ulonglong4(tmp[lid][counterBase], tmp[lid][counterBase + 1], tmp[lid][counterBase + 2], tmp[lid][counterBase + 3]);
			destination[idx + 1] = make_ulonglong4(tmp[lid][counterBase + 4], tmp[lid][counterBase + 5], tmp[lid][counterBase + 6], tmp[lid][counterBase + 7]);
		}
	}
	__global__  void FluffyRound(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		const int edgesInBucket = min(sourceIndexes[group], bktInSize);
		const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

		for (int i = 0; i < 16; i++)
			ecounters[lid + (512 * i)] = 0;

		__syncthreads();

		for (int i = 0; i < loops; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucket)
			{
				const int index = (bktInSize * group) + lindex;

				uint2 edge = source[index];

				if (edge.x == 0 && edge.y == 0) continue;

				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}

		__syncthreads();

		for (int i = 0; i < loops; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucket)
			{
				const int index = (bktInSize * group) + lindex;

				uint2 edge = __ldg(&source[index]);

				if (edge.x == 0 && edge.y == 0) continue;

				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}

	}
	__global__  void FluffyTail(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		int myEdges = sourceIndexes[group];
		__shared__ int destIdx;

		if (lid == 0)
			destIdx = atomicAdd(destinationIndexes, myEdges);

		__syncthreads();

		if (lid < myEdges)
		{
			destination[destIdx + lid] = source[group * DUCK_B_EDGES / 4 + lid];
		}
	}
	__global__  void FluffyRecovery(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, int * indexes)
	{
		const int gid = blockDim.x * blockIdx.x + threadIdx.x;
		const int lid = threadIdx.x;

		__shared__ u32 nonces[42];
		u64 sipblock[64];

		uint64_t v0;
		uint64_t v1;
		uint64_t v2;
		uint64_t v3;

		if (lid < 42) nonces[lid] = 0;

		__syncthreads();

		for (int i = 0; i < 1024; i += EDGE_BLOCK_SIZE)
		{
			u64 blockNonce = gid * 1024 + i;

			v0 = v0i;
			v1 = v1i;
			v2 = v2i;
			v3 = v3i;

			for (u32 b = 0; b < EDGE_BLOCK_SIZE; b++)
			{
				v3 ^= blockNonce + b;
				SIPROUND; SIPROUND;
				v0 ^= blockNonce + b;
				v2 ^= 0xff;
				SIPROUND; SIPROUND; SIPROUND; SIPROUND;

				sipblock[b] = (v0 ^ v1) ^ (v2  ^ v3);

			}
			const u64 last = sipblock[EDGE_BLOCK_MASK];

			for (short s = EDGE_BLOCK_MASK; s >= 0; s--)
			{
				u64 lookup = s == EDGE_BLOCK_MASK ? last : sipblock[s] ^ last;
				u64 u = lookup & EDGEMASK;
				u64 v = (lookup >> 32) & EDGEMASK;

				u64 a = u | (v << 32);
				u64 b = v | (u << 32);

				for (int i = 0; i < 42; i++)
				{
					if ((recovery[i] == a) || (recovery[i] == b))
						nonces[i] = blockNonce + s;
				}
			}
		}

		__syncthreads();

		if (lid < 42)
		{
			if (nonces[lid] > 0)
				indexes[lid] = nonces[lid];
		}
	}

}

