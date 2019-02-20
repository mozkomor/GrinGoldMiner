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


#define DUCK_SIZE_A 134LL
#define DUCK_SIZE_B 84LL

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
#define BKTGRAN 64

#define EDGECNT 562036736
#define BUKETS 4096
#define BUKET_MASK (BUKETS-1)
#define BUKET_SIZE (EDGECNT/BUKETS)

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  { \
    v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
    v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
    v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
    v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
  }
#define SIPBLOCK(b) \
	{\
		v3 ^= blockNonce + b;\
		for (short r = 0; r < 2; r++)\
		SIPROUND;\
		v0 ^= blockNonce + b;\
		v2 ^= 0xff;\
		for (short r = 0; r < 4; r++)\
		SIPROUND;\
	}
#define DUMP(E)\
{\
	u64 lookup = E;\
	uint2 edge1 = make_uint2( lookup & EDGEMASK, (lookup >> 32) & EDGEMASK );\
	int bucket = edge1.x & BUKET_MASK;\
	\
	u64 stEdge = atomicExch(&magazine[bucket], (u64)0);\
	if (stEdge == 0)\
	{\
		u64 edge64 = (((u64)edge1.y) << 32) | edge1.x;\
		u64 res = atomicCAS(&magazine[bucket], 0, edge64);\
		if (res != 0)\
		{\
			int block = bucket / 1024;\
			int shift = (BUKET_SIZE / 4 * 4096) * block;\
			int position = (min(BUKET_SIZE / 4 - 4, (atomicAdd(indexes + bucket, 2))));\
			int idx = (shift+((bucket%1024) * (BUKET_SIZE / 4) + position)) / 2;\
			buffer[idx] = make_uint4(edge64, edge64 >> 32, 0, 0);\
		}\
	}\
	else\
	{\
		int block = bucket / 1024;\
		int shift = (BUKET_SIZE / 4 * 4096) * block;\
		int position = (min(BUKET_SIZE / 4 - 4, (atomicAdd(indexes + bucket, 2))));\
		int idx = (shift+((bucket%1024) * (BUKET_SIZE / 4) + position)) / 2;\
		buffer[idx] = make_uint4(stEdge, stEdge >> 32, edge1.x, edge1.y);\
	}\
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

	__global__  void FluffySeed4K(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, uint4 * __restrict__ buffer, int * __restrict__ indexes, const int offset)
	{
		const int gid = blockDim.x * blockIdx.x + threadIdx.x;
		const int lid = threadIdx.x;

		ulonglong4 sipblockL[15];
		__shared__ u64 magazine[4096];

		uint64_t v0, v1, v2, v3;

		for (short i = 0; i < 8; i++)
			magazine[lid + (512 * i)] = 0;

		__syncthreads();

		for (short i = 0; i < 256; i += EDGE_BLOCK_SIZE)
		{
			u64 blockNonce = offset + (gid * 256 + i);

			v0 = v0i;
			v1 = v1i;
			v2 = v2i;
			v3 = v3i;

			for (short b = 0; b < 60; b += 4)
			{
				SIPBLOCK(b);
				u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
				SIPBLOCK(b + 1);
				u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
				SIPBLOCK(b + 2);
				u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
				SIPBLOCK(b + 3);
				u64 e4 = (v0 ^ v1) ^ (v2  ^ v3);
				sipblockL[b / 4] = make_ulonglong4(e1, e2, e3, e4);
			}

			SIPBLOCK(60);
			u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
			SIPBLOCK(61);
			u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
			SIPBLOCK(62);
			u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
			SIPBLOCK(63);
			u64 last = (v0 ^ v1) ^ (v2  ^ v3);

			DUMP(last);
			DUMP(e1 ^ last);
			DUMP(e2 ^ last);
			DUMP(e3 ^ last);

			for (short s = 14; s >= 0; s--)
			{
				ulonglong4 edges = sipblockL[s];
				DUMP(edges.x ^ last);
				DUMP(edges.y ^ last);
				DUMP(edges.z ^ last);
				DUMP(edges.w ^ last);
			}
		}

		__syncthreads();

		for (int i = 0; i < 8; i++)
		{
			int bucket = lid + (512 * i);
			u64 edge = magazine[bucket];
			if (edge != 0)
			{
				int block = bucket / 1024;
				int shift = (BUKET_SIZE / 4 * 4096) * block;
				int position = (min(BUKET_SIZE / 4 - 4, (atomicAdd(indexes + bucket, 2))));
				int idx = (shift + ((bucket%1024) * (BUKET_SIZE / 4) + position)) / 2;
				buffer[idx] = make_uint4(edge, edge >> 32, 0, 0);
			}
		}
	}
	__global__  void FluffyRound_S(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize, const int offset)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		for (int i = 0; i < 16; i++)
			ecounters[lid + (512 * i)] = 0;

		const int edgesInBucket1 = min(sourceIndexes[group + offset], BUKET_SIZE / 4 - 4);
		const int loops1 = (edgesInBucket1 + CTHREADS) / CTHREADS;
		const int edgesInBucket2 = min(sourceIndexes[group + 4096 + offset], BUKET_SIZE / 4 - 4);
		const int loops2 = (edgesInBucket2 + CTHREADS) / CTHREADS;
		const int edgesInBucket3 = min(sourceIndexes[group + 8192 + offset], BUKET_SIZE / 4 - 4);
		const int loops3 = (edgesInBucket3 + CTHREADS) / CTHREADS;
		const int edgesInBucket4 = min(sourceIndexes[group + 12288 + offset], BUKET_SIZE / 4 - 4);
		const int loops4 = (edgesInBucket4 + CTHREADS) / CTHREADS;

		const uint2 * source1 = source + (((int)BUKET_SIZE / 4) * (group + (offset << 2)));
		const uint2 * source2 = source + (((int)BUKET_SIZE / 4) * (group + 1024 + (offset << 2)));
		const uint2 * source3 = source + (((int)BUKET_SIZE / 4) * (group + 2048 + (offset << 2)));
		const uint2 * source4 = source + (((int)BUKET_SIZE / 4) * (group + 3072 + (offset << 2)));

		const int bktOffset = offset * 4;

		__syncthreads();

		for (int i = 0; i < loops1; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket1)
			{
				uint2 edge = source1[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}
		for (int i = 0; i < loops2; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket2)
			{
				uint2 edge = source2[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}
		for (int i = 0; i < loops3; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket3)
			{
				uint2 edge = source3[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}
		for (int i = 0; i < loops4; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket4)
			{
				uint2 edge = source4[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}

		__syncthreads();

		for (int i = 0; i < loops1; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket1)
			{
				uint2 edge = source1[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}
		for (int i = 0; i < loops2; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket2)
			{
				uint2 edge = source2[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}
		for (int i = 0; i < loops3; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket3)
			{
				uint2 edge = source3[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}
		for (int i = 0; i < loops4; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket4)
			{
				uint2 edge = source4[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}

	}
	__global__  void FluffyRound(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		const int edgesInBucket = min(sourceIndexes[group], bktInSize);
		const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

		const int offset = (bktInSize * group);

		for (int i = 0; i < 16; i++)
			ecounters[lid + (512 * i)] = 0;

		__syncthreads();

		for (int i = 0; i < loops; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket)
			{
				uint2 edge = __ldg(&source[offset+lindex]);
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}

		__syncthreads();

		for (int i = 0; i < loops; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket)
			{
				uint2 edge = __ldg(&source[offset+lindex]);
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}

	}
	__global__  void FluffyRound_J(const uint2 * sourceA, const uint2 * sourceB, const uint2 * sourceC, const uint2 * sourceD, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		for (int i = 0; i < 16; i++)
			ecounters[lid + (512 * i)] = 0;

		const int edgesInBucketA = min(sourceIndexes[group], bktInSize);
		const int edgesInBucketB = min(sourceIndexes[group + 4096], bktInSize);
		const int edgesInBucketC = min(sourceIndexes[group + 8192], bktInSize);
		const int edgesInBucketD = min(sourceIndexes[group + 12288], bktInSize);

		const int loopsA = (edgesInBucketA + CTHREADS) / CTHREADS;
		const int loopsB = (edgesInBucketB + CTHREADS) / CTHREADS; 
		const int loopsC = (edgesInBucketC + CTHREADS) / CTHREADS;
		const int loopsD = (edgesInBucketD + CTHREADS) / CTHREADS;

		const int offset = (bktInSize * group);

		__syncthreads();

		for (int i = 0; i < loopsA; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketA)
			{
				uint2 edge = sourceA[offset+lindex];
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}
		for (int i = 0; i < loopsB; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketB)
			{
				uint2 edge = sourceB[offset+lindex];
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}
		for (int i = 0; i < loopsC; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketC)
			{
				uint2 edge = sourceC[offset+lindex];
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}
		for (int i = 0; i < loopsD; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketD)
			{
				uint2 edge = sourceD[offset+lindex];
				Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
			}
		}

		__syncthreads();

		for (int i = 0; i < loopsA; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketA)
			{
				uint2 edge = sourceA[offset+lindex];
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}
		for (int i = 0; i < loopsB; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketB)
			{
				uint2 edge = sourceB[offset+lindex];
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}
		for (int i = 0; i < loopsC; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketC)
			{
				uint2 edge = sourceC[offset + lindex];
				if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
				{
					const int bucket = edge.y & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}
		for (int i = 0; i < loopsD; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketD)
			{
				uint2 edge = sourceD[offset + lindex];
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

