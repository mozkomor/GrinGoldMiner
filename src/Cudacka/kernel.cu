// Cuckarood Cycle, a memory-hard proof-of-work by John Tromp and team Grin
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
#define DUCK_SIZE_B 86LL

#define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL)

#define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL)

#define EDGE_BLOCK_SIZE (64)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)

#define EDGEBITS 29
#define NEDGES2 ((node_t)1 << EDGEBITS)
#define NEDGES1 (NEDGES2/2)
#define NNODES1 NEDGES1
#define NNODES2 NEDGES2

#define EDGEMASK (NEDGES2 - 1)
#define NODE1MASK (NNODES1 - 1)

#define CTHREADS 1024
#define CTHREADS512 512
#define BKTMASK4K (4096-1)
#define BKTGRAN 64

#define EDGECNT 562036736
#define BUKETS 4096
#define BUKET_MASK (BUKETS-1)
#define BUKET_SIZE (EDGECNT/BUKETS)

#define XBITS 6
const u32 NX = 1 << XBITS;
const u32 NX2 = NX * NX;
const u32 XMASK = NX - 1;
const u32 YBITS = XBITS;
const u32 NY = 1 << YBITS;
const u32 YZBITS = EDGEBITS - XBITS;
const u32 ZBITS = YZBITS - YBITS;
const u32 NZ = 1 << ZBITS;
const u32 ZMASK = NZ - 1;

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  { \
    v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
    v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
    v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
    v1 = ROTL(v1,17);   v3 = ROTL(v3,25); \
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
#define DUMP(E, dir)\
{\
	u64 lookup = E;\
	uint2 edge1 = make_uint2( (lookup & NODE1MASK) << 1 | dir, (lookup >> 31) & (NODE1MASK << 1) | dir);\
	int bucket = (edge1.x >> ZBITS) & BUKET_MASK;\
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

__device__ __forceinline__ uint2 ld_cs_u32_v2(const uint2 *p_src)
{
	uint2 n_result;
	asm("ld.global.cs.v2.u32 {%0,%1}, [%2];"  : "=r"(n_result.x), "=r"(n_result.y) : "l"(p_src));
	return n_result;
}

__device__ __forceinline__ void st_cg_u32_v2(uint2 *p_dest, const uint2 n_value)
{
	asm("st.global.cg.v2.u32 [%0], {%1, %2};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y));
}

__device__ __forceinline__ void st_cg_u32_v4(uint4 *p_dest, const uint4 n_value)
{
	asm("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y), "r"(n_value.z), "r"(n_value.w));
}

__device__ __forceinline__  void Increase2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;
	atomicOr(ecounters + word, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;
	return (ecounters[word] >> bit) & 1;
}

#define ST_RS \
{\
	int bucket = (edge.y >> ZBITS) & BKTMASK4K;\
	u64 edge2 = atomicExch(&magazine[bucket], (u64)0);\
	if (edge2 == 0)\
	{\
		u64 res = atomicCAS(&magazine[bucket], 0, (((u64)edge.y) << 32) | (u64)edge.x);\
		if (res != 0)\
		{\
			int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 2), bktOutSize - 4);\
			destination[ ((bucket * bktOutSize) + bktIdx) / 2] = make_uint4(edge.y, edge.x, 0, 0);\
		}\
	}\
	else\
	{\
		int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 2), bktOutSize - 4);\
		destination[ ((bucket * bktOutSize) + bktIdx) / 2] = make_uint4(edge.y, edge.x, edge2 >> 32, edge2);\
	}\
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

			DUMP(last,      1);
			DUMP(e1 ^ last, 0);
			DUMP(e2 ^ last, 1);
			DUMP(e3 ^ last, 0);

			for (short s = 14; s >= 0; s--)
			{
				ulonglong4 edges = sipblockL[s];
				DUMP(edges.x ^ last, 0);
				DUMP(edges.y ^ last, 1);
				DUMP(edges.z ^ last, 0);
				DUMP(edges.w ^ last, 1);
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
				int idx = (shift + ((bucket % 1024) * (BUKET_SIZE / 4) + position)) / 2;
				buffer[idx] = make_uint4(edge, edge >> 32, 0, 0);
			}
		}
	}
	__global__  void FluffyRound_A1(const uint2 * source, uint4 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize, const int offset)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192 + 4096];
		u64 * magazine = (u64 *)(ecounters + 4096);

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

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
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops2; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket2)
			{
				uint2 edge = source2[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops3; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket3)
			{
				uint2 edge = source3[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops4; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket4)
			{
				uint2 edge = source4[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}

		__syncthreads();

		for (int i = 0; i < 4; i++)
			magazine[lid + (1024 * i)] = 0;

		__syncthreads();

		for (int i = 0; i < loops1; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket1)
			{
				uint2 edge = source1[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
				}
			}
		}

		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			int bucket = lid + (1024 * i);
			u64 edge = magazine[bucket];
			if (edge != 0)
			{
				int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 2), bktOutSize - 4);
				destination[((bucket * bktOutSize) + bktIdx) / 2] = make_uint4(edge >> 32, edge, 0, 0);
			}
		}

	}

	#define THREADS_A2 512
	__global__  void FluffyRound_A2(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize, const int round, int * aux)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[4096];

		const int edgesInBucket = min(sourceIndexes[group], bktInSize);
		const int loops = (edgesInBucket + THREADS_A2) / THREADS_A2;

		const int offset = (bktInSize * group);

		for (int i = 0; i < 8; i++)
			ecounters[lid + (THREADS_A2 * i)] = 0;

		__syncthreads();

		if (loops > 1)
		{
			for (int i = 0; i < loops - 1; i++)
			{
				const int lindex = (i * THREADS_A2) + lid;
				if (lindex < edgesInBucket)
				{
					uint2 edge = source[offset + lindex];
					Increase2bCounter(ecounters, (edge.x & ZMASK));
				}
			}

			uint2 edge1;
			int lindex = ((loops - 1) * THREADS_A2) + lid;
			if (lindex < edgesInBucket)
			{
				edge1 = ld_cs_u32_v2(&source[offset + lindex]);
				Increase2bCounter(ecounters, (edge1.x & ZMASK));
			}

			__syncthreads();

			if (lindex < edgesInBucket)
			{
				if (Read2bCounter(ecounters, (edge1.x & ZMASK) ^ 1))
				{
					const int bucket = (edge1.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge1.y, edge1.x));
				}
			}

			for (int i = loops - 2; i >= 0; i--)
			{
				const int lindex = (i * THREADS_A2) + lid;
				if (lindex < edgesInBucket)
				{
					uint2 edge = source[offset + lindex];
					if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
					{
						const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
						const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
						st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
					}
				}
			}
		}
		else
		{
			uint2 edge1;
			int lindex = lid;

			if (lindex < edgesInBucket)
			{
				edge1 = ld_cs_u32_v2(&source[offset + lindex]);
				Increase2bCounter(ecounters, (edge1.x & ZMASK));
			}

			__syncthreads();

			if (lindex < edgesInBucket)
			{
				if (Read2bCounter(ecounters, (edge1.x & ZMASK) ^ 1))
				{
					const int bucket = (edge1.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge1.y, edge1.x));
				}
			}
		}


		//__syncthreads();

		//if (lid == 0)
		//{
		//	ecounters[group] = atomicAdd(aux + round, 1);
		//}

		//__syncthreads();

		//if (ecounters[group] == 4095)
		//{
		//	for (int i = 0; i < 4096 / CTHREADS512; i++)
		//		((int*)sourceIndexes)[lid + (CTHREADS512 * i)] = 0;
		//}


	}
	__global__  void FluffyRound_A3(const uint2 * sourceA, const uint2 * sourceB, const uint2 * sourceC, const uint2 * sourceD, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

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
				uint2 edge = ld_cs_u32_v2(&sourceA[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsB; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketB)
			{
				uint2 edge = ld_cs_u32_v2(&sourceB[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsC; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketC)
			{
				uint2 edge = ld_cs_u32_v2(&sourceC[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsD; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketD)
			{
				uint2 edge = ld_cs_u32_v2(&sourceD[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}

		__syncthreads();

		for (int i = 0; i < loopsA; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketA)
			{
				uint2 edge = ld_cs_u32_v2(&sourceA[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
		for (int i = 0; i < loopsB; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketB)
			{
				uint2 edge = ld_cs_u32_v2(&sourceB[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
		for (int i = 0; i < loopsC; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketC)
			{
				uint2 edge = ld_cs_u32_v2(&sourceC[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
		for (int i = 0; i < loopsD; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketD)
			{
				uint2 edge = ld_cs_u32_v2(&sourceD[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
	}

	__global__  void FluffySeed4K_C0(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, uint4 * __restrict__ buffer, int * __restrict__ indexes, const int offset)
	{
		const int gid = blockDim.x * blockIdx.x + threadIdx.x;
		const int lid = threadIdx.x;

		ulonglong4 sipblockL[15];
		__shared__ u64 magazine[4096];

		uint64_t v0, v1, v2, v3;

		for (short i = 0; i < 4; i++)
			magazine[lid + (1024 * i)] = 0;

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

			DUMP(last, 1);
			DUMP(e1 ^ last, 0);
			DUMP(e2 ^ last, 1);
			DUMP(e3 ^ last, 0);

			for (short s = 14; s >= 0; s--)
			{
				ulonglong4 edges = sipblockL[s];
				DUMP(edges.x ^ last, 0);
				DUMP(edges.y ^ last, 1);
				DUMP(edges.z ^ last, 0);
				DUMP(edges.w ^ last, 1);
			}
		}

		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			int bucket = lid + (1024 * i);
			u64 edge = magazine[bucket];
			if (edge != 0)
			{
				int block = bucket / 1024;
				int shift = (BUKET_SIZE / 4 * 4096) * block;
				int position = (min(BUKET_SIZE / 4 - 4, (atomicAdd(indexes + bucket, 2))));
				int idx = (shift + ((bucket % 1024) * (BUKET_SIZE / 4) + position)) / 2;
				buffer[idx] = make_uint4(edge, edge >> 32, 0, 0);
			}
		}
	}
	__global__  void FluffyRound_C1(const uint2 * source, uint4 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize, const int offset)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192 + 4096];
		u64 * magazine = (u64 *)(ecounters + 4096);

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

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
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops2; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket2)
			{
				uint2 edge = source2[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops3; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket3)
			{
				uint2 edge = source3[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops4; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket4)
			{
				uint2 edge = source4[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}

		__syncthreads();

		for (int i = 0; i < 4; i++)
			magazine[lid + (1024 * i)] = 0;

		__syncthreads();

		for (int i = 0; i < loops1; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket1)
			{
				uint2 edge = source1[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					ST_RS;
				}
			}
		}

		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			int bucket = lid + (1024 * i);
			u64 edge = magazine[bucket];
			if (edge != 0)
			{
				int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 2), bktOutSize - 4);
				destination[((bucket * bktOutSize) + bktIdx) / 2] = make_uint4(edge >> 32, edge, 0, 0);
			}
		}

	}
	__global__  void FluffyRound_C2(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize, const int round, int * aux)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		const int edgesInBucket = min(sourceIndexes[group], bktInSize);
		const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

		const int offset = (bktInSize * group);

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

		__syncthreads();

		if (loops > 1)
		{
			for (int i = 0; i < loops - 1; i++)
			{
				const int lindex = (i * CTHREADS) + lid;
				if (lindex < edgesInBucket)
				{
					uint2 edge = source[offset + lindex];
					Increase2bCounter(ecounters, (edge.x & ZMASK));
				}
			}

			uint2 edge1;
			int lindex = ((loops - 1) * CTHREADS) + lid;
			if (lindex < edgesInBucket)
			{
				edge1 = ld_cs_u32_v2(&source[offset + lindex]);
				Increase2bCounter(ecounters, (edge1.x & ZMASK));
			}

			__syncthreads();

			if (lindex < edgesInBucket)
			{
				if (Read2bCounter(ecounters, (edge1.x & ZMASK) ^ 1))
				{
					const int bucket = (edge1.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge1.y, edge1.x));
				}
			}

			for (int i = loops - 2; i >= 0; i--)
			{
				const int lindex = (i * CTHREADS) + lid;
				if (lindex < edgesInBucket)
				{
					uint2 edge = source[offset + lindex];
					if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
					{
						const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
						const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
						st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
					}
				}
			}
		}
		else
		{
			uint2 edge1;
			int lindex = lid;

			if (lindex < edgesInBucket)
			{
				edge1 = ld_cs_u32_v2(&source[offset + lindex]);
				Increase2bCounter(ecounters, (edge1.x & ZMASK));
			}

			__syncthreads();

			if (lindex < edgesInBucket)
			{
				if (Read2bCounter(ecounters, (edge1.x & ZMASK) ^ 1))
				{
					const int bucket = (edge1.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge1.y, edge1.x));
				}
			}
		}


		//__syncthreads();

		//if (lid == 0)
		//{
		//	ecounters[group] = atomicAdd(aux + round, 1);
		//}

		//__syncthreads();

		//if (ecounters[group] == 4095)
		//{
		//	for (int i = 0; i < 4096 / CTHREADS512; i++)
		//		((int*)sourceIndexes)[lid + (CTHREADS512 * i)] = 0;
		//}


	}
	__global__  void FluffyRound_C3(const uint2 * sourceA, const uint2 * sourceB, const uint2 * sourceC, const uint2 * sourceD, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

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
				uint2 edge = ld_cs_u32_v2(&sourceA[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsB; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketB)
			{
				uint2 edge = ld_cs_u32_v2(&sourceB[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsC; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketC)
			{
				uint2 edge = ld_cs_u32_v2(&sourceC[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK) );
			}
		}
		for (int i = 0; i < loopsD; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketD)
			{
				uint2 edge = ld_cs_u32_v2(&sourceD[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}

		__syncthreads();

		for (int i = 0; i < loopsA; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketA)
			{
				uint2 edge = ld_cs_u32_v2(&sourceA[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
		for (int i = 0; i < loopsB; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketB)
			{
				uint2 edge = ld_cs_u32_v2(&sourceB[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
		for (int i = 0; i < loopsC; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketC)
			{
				uint2 edge = ld_cs_u32_v2(&sourceC[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
		for (int i = 0; i < loopsD; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketD)
			{
				uint2 edge = ld_cs_u32_v2(&sourceD[offset + lindex]);
				if (edge.x == 0 && edge.y == 0) continue;
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 2);
					st_cg_u32_v2(&destination[(bucket * bktOutSize) + bktIdx], make_uint2(edge.y, edge.x));
				}
			}
		}
	}

	__global__  void FluffyRound_B1(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize, const int offset)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

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
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops2; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket2)
			{
				uint2 edge = source2[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops3; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket3)
			{
				uint2 edge = source3[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loops4; i++)
		{
			int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket4)
			{
				uint2 edge = source4[lindex];
				if (edge.x == 0 && edge.y == 0) continue;
				Increase2bCounter(ecounters, (edge.x & ZMASK));
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket + bktOffset, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}

	}
	__global__  void FluffyRound_B2(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize, const int round, int * aux)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		const int edgesInBucket = min(sourceIndexes[group], bktInSize);
		const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

		const int offset = (bktInSize * group);

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

		__syncthreads();

		for (int i = 0; i < loops; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket)
			{
				uint2 edge = __ldg(&source[offset + lindex]);
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}

		__syncthreads();

		for (int i = 0; i < loops; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucket)
			{
				uint2 edge = __ldg(&source[offset + lindex]);
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
					const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
					destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
				}
			}
		}

		__syncthreads();

		if (lid == 0)
		{
			ecounters[group] = atomicAdd(aux + round, 1);
		}

		__syncthreads();

		if (ecounters[group] == 4095)
		{
			for (int i = 0; i < 4; i++)
				((int*)sourceIndexes)[lid + (1024 * i)] = 0;
		}


	}
	__global__  void FluffyRound_B3(const uint2 * sourceA, const uint2 * sourceB, const uint2 * sourceC, const uint2 * sourceD, uint2 * destination, const int * sourceIndexes, int * destinationIndexes, const int bktInSize, const int bktOutSize)
	{
		const int lid = threadIdx.x;
		const int group = blockIdx.x;

		__shared__ u32 ecounters[8192];

		for (int i = 0; i < 8; i++)
			ecounters[lid + (1024 * i)] = 0;

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
				uint2 edge = sourceA[offset + lindex];
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsB; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketB)
			{
				uint2 edge = sourceB[offset + lindex];
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsC; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketC)
			{
				uint2 edge = sourceC[offset + lindex];
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}
		for (int i = 0; i < loopsD; i++)
		{
			const int lindex = (i * CTHREADS) + lid;
			if (lindex < edgesInBucketD)
			{
				uint2 edge = sourceD[offset + lindex];
				Increase2bCounter(ecounters, (edge.x & ZMASK));
			}
		}

		__syncthreads();

		for (int i = 0; i < loopsA; i++)
		{
			const int lindex = (i * CTHREADS) + lid;

			if (lindex < edgesInBucketA)
			{
				uint2 edge = sourceA[offset + lindex];
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
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
				uint2 edge = sourceB[offset + lindex];
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
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
				if (Read2bCounter(ecounters, (edge.x & ZMASK) ^ 1))
				{
					const int bucket = (edge.y >> ZBITS) & BKTMASK4K;
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
				u32 dir = s & 1;
				u64 lookup = s == EDGE_BLOCK_MASK ? last : sipblock[s] ^ last;

				u64 u = (lookup & NODE1MASK) << 1 | dir;
				u64 v = (lookup >> 31) & (NODE1MASK << 1) | dir;

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

