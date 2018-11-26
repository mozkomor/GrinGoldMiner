// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Jiri Vadura - photon
// This CUDA part of Theta optimized miner is covered by the FAIR MINING license

#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <xmmintrin.h>
#include <algorithm>
#include <stdio.h>
#include <stdint.h>
#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u32 node_t;
typedef u64 nonce_t;

	#ifdef _WIN32
	#define DUCK_SIZE_A 129LL
	#define DUCK_SIZE_B 82LL
	#else
	#define DUCK_SIZE_A 130LL
	#define DUCK_SIZE_B 85LL
	#endif


#define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL)

#define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL)

#define EDGE_BLOCK_SIZE (64)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)

#define EDGEBITS 29
// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK (NEDGES - 1)

#define CTHREADS 1024
#define BKTMASK4K (4096-1)

__constant__ u64 recovery[42];

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
    v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
    v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
    v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
  } while(0)


__device__  node_t dipnode(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const  nonce_t nce, const  u32 uorv) {
	u64 nonce = 2 * nce + uorv;
	u64 v0 = v0i, v1 = v1i, v2 = v2i, v3 = v3i ^ nonce;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

__device__ ulonglong4 Pack4edges(const uint2 e1, const  uint2 e2, const  uint2 e3, const  uint2 e4)
{
	u64 r1 = (((u64)e1.y << 32) | ((u64)e1.x));
	u64 r2 = (((u64)e2.y << 32) | ((u64)e2.x));
	u64 r3 = (((u64)e3.y << 32) | ((u64)e3.x));
	u64 r4 = (((u64)e4.y << 32) | ((u64)e4.x));
	return make_ulonglong4(r1, r2, r3, r4);
}

__global__  void FluffyRecovery(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, ulonglong4 * buffer, int * indexes)
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

	for (int i = 0; i < 1024 * 4; i += EDGE_BLOCK_SIZE)
	{
		u64 blockNonce = gid * (1024 * 4) + i;

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
			u32 u = lookup & EDGEMASK;
			u32 v = (lookup >> 32) & EDGEMASK;

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

__global__  void FluffySeed2A(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, ulonglong4 * buffer, int * indexes)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;

	__shared__ uint2 tmp[64][15];
	__shared__ int counters[64];
	u64 sipblock[64];

	uint64_t v0;
	uint64_t v1;
	uint64_t v2;
	uint64_t v3;

	counters[lid] = 0;

	__syncthreads();

	for (int i = 0; i < 1024 * 16; i += EDGE_BLOCK_SIZE)
	{
		u64 blockNonce = gid * (1024 * 16) + i;

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
			uint2 hash = make_uint2(lookup & EDGEMASK, (lookup >> 32) & EDGEMASK);
			int bucket = hash.x & 63;

			__syncthreads();

			int counter = min((int)atomicAdd(counters + bucket, 1), (int)14);

			tmp[bucket][counter] = hash;

			__syncthreads();

			{
				int localIdx = min(15, counters[lid]);

				if (localIdx >= 8)
				{
					int newCount = (localIdx - 8);
					counters[lid] = newCount;

					{
						int cnt = min((int)atomicAdd(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));

						{
							buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
							buffer[(lid * DUCK_A_EDGES_64 + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
						}
					}

					for (int t = 0; t < newCount; t++)
					{
						tmp[lid][t] = tmp[lid][t + 8];
					}

				}
			}
		}
	}

	__syncthreads();

	{
		int localIdx = min(16, counters[lid]);

		if (localIdx > 0)
		{
			int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
			buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(
				tmp[lid][0],
				localIdx > 1 ? tmp[lid][1] : make_uint2(0, 0),
				localIdx > 2 ? tmp[lid][2] : make_uint2(0, 0),
				localIdx > 3 ? tmp[lid][3] : make_uint2(0, 0));
		}
		if (localIdx > 4)
		{
			int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
			buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(
				tmp[lid][4],
				localIdx > 5 ? tmp[lid][5] : make_uint2(0, 0),
				localIdx > 6 ? tmp[lid][6] : make_uint2(0, 0),
				localIdx > 7 ? tmp[lid][7] : make_uint2(0, 0));
		}
	}

}

#define BKTGRAN 32
__global__  void FluffySeed2B(const  uint2 * source, ulonglong4 * destination, const  int * sourceIndexes, int * destinationIndexes, int startBlock)
{
	//const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	__shared__ uint2 tmp[64][15];
	__shared__ int counters[64];

	counters[lid] = 0;

	__syncthreads();

	const int offsetMem = startBlock * DUCK_A_EDGES_64;
	const int myBucket = group / BKTGRAN;
	const int microBlockNo = group % BKTGRAN;
	const int bucketEdges = min(sourceIndexes[myBucket + startBlock], (int)(DUCK_A_EDGES_64));
	const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN);
	const int loops = (microBlockEdgesCount / 64);

	for (int i = 0; i < loops; i++)
	{
		int edgeIndex = (microBlockNo * microBlockEdgesCount) + (64 * i) + lid;

		if (edgeIndex < bucketEdges)
		{
			uint2 edge = source[offsetMem + (myBucket * DUCK_A_EDGES_64) + edgeIndex];

			if (edge.x == 0 && edge.y == 0) continue;

			int bucket = (edge.x >> 6) & (64 - 1);

			__syncthreads();

			int counter = min((int)atomicAdd(counters + bucket, 1), (int)14);

			tmp[bucket][counter] = edge;

			__syncthreads();

			int localIdx = min(15, counters[lid]);

			if (localIdx >= 8)
			{
				int newCount = (localIdx - 8);
				counters[lid] = newCount;

				{
					int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), (int)(DUCK_A_EDGES - 8));

					{
						destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
						destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
					}
				}

				for (int t = 0; t < newCount; t++)
				{
					tmp[lid][t] = tmp[lid][t + 8];
				}

			}
		}
	}

	__syncthreads();

	{
		int localIdx = min(16, counters[lid]);

		if (localIdx > 0)
		{
			int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 4), (int)(DUCK_A_EDGES - 4));
			destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(
				tmp[lid][0],
				localIdx > 1 ? tmp[lid][1] : make_uint2(0, 0),
				localIdx > 2 ? tmp[lid][2] : make_uint2(0, 0),
				localIdx > 3 ? tmp[lid][3] : make_uint2(0, 0));
		}
		if (localIdx > 4)
		{
			int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 4), (int)(DUCK_A_EDGES - 4));
			destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(
				tmp[lid][4],
				localIdx > 5 ? tmp[lid][5] : make_uint2(0, 0),
				localIdx > 6 ? tmp[lid][6] : make_uint2(0, 0),
				localIdx > 7 ? tmp[lid][7] : make_uint2(0, 0));
		}
	}
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

template<int bktInSize, int bktOutSize>
__global__   void FluffyRound(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
{
	//const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	__shared__ u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

	ecounters[lid] = 0;
	ecounters[lid + 1024] = 0;
	ecounters[lid + (1024 * 2)] = 0;
	ecounters[lid + (1024 * 3)] = 0;
	ecounters[lid + (1024 * 4)] = 0;
	ecounters[lid + (1024 * 5)] = 0;
	ecounters[lid + (1024 * 6)] = 0;
	ecounters[lid + (1024 * 7)] = 0;

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

			uint2 edge = source[index];

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

template __global__ void FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES / 2>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 2>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 4>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES / 4, DUCK_B_EDGES / 4>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);



__global__   void /*Magical*/FluffyTail/*Pony*/(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
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

//static u32 hostB[2 * 260000];
static u64 h_mydata[42];

int main(int argc, char* argv[])
{
	int device = 0;
	if (argc >= 2)
		device = atoi(argv[1]);

	std::ofstream myfile;

	u32 * buffer = new u32[150000 * 2];

	const size_t bufferSize = DUCK_SIZE_A * 1024 * 4096 * 8;
	const size_t bufferSize2 = DUCK_SIZE_B * 1024 * 4096 * 8;
	const size_t indexesSize = 128 * 128 * 4;

	//const unsigned int edges = (1 << 29);

	u32 * bufferA;
	u32 * bufferB;
	u32 * indexesE;
	u32 * indexesE2;

	u32 hostA[256 * 256];

	cudaError_t cudaStatus;
	size_t free_device_mem = 0;
	size_t total_device_mem = 0;

	unsigned long long k0 = 0xa34c6a2bdaa03a14ULL;
	unsigned long long k1 = 0xd736650ae53eee9eULL;
	unsigned long long k2 = 0x9a22f05e3bffed5eULL;
	unsigned long long k3 = 0xb8d55478fa3a606dULL;

	unsigned long long nonce = 0;

#ifdef _WIN32
	LPCWSTR CDS = L"CDS0";

	switch (device)
	{
	case 1:
		CDS = L"CDS1"; break;
	case 2:
		CDS = L"CDS2"; break;
	case 3:
		CDS = L"CDS3"; break;
	case 4:
		CDS = L"CDS4"; break;
	case 5:
		CDS = L"CDS5"; break;
	case 6:
		CDS = L"CDS6"; break;
	}

	HANDLE handle = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, 8000000, CDS);
	u32 * sharedData = (u32*)MapViewOfFile(handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 8000000);
#else

#endif


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
	
	cudaMemGetInfo(&free_device_mem, &total_device_mem);

	fprintf(stderr, "Currently available amount of device memory: %zu bytes\n", free_device_mem);
	fprintf(stderr, "Total amount of device memory: %zu bytes\n", total_device_mem);

	cudaStatus = cudaMalloc((void**)&bufferA, bufferSize+bufferSize2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "status: %s\n", cudaGetErrorString(cudaStatus));
		fprintf(stderr, "cudaMalloc failed buffer A 4GB!\n");
		goto Error;
	}

	fprintf(stderr, "Allociating buffer 1+2\n");

	cudaMemGetInfo(&free_device_mem, &total_device_mem);

	u32 * bufferMid = bufferA + (bufferSize2 / 4);
	bufferB = bufferA + (bufferSize / 4);
	
	cudaStatus = cudaMalloc((void**)&indexesE, indexesSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed Index array 1!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&indexesE2, indexesSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed Index array 2!\n");
		goto Error;
	}

	cudaMemGetInfo(&free_device_mem, &total_device_mem);

	fprintf(stderr, "Currently available amount of device memory: %zu bytes\n", free_device_mem);

	fprintf(stderr, "CUDA device armed\n");

	// loop starts here
	// wait for header hashes, nonce+r

	while (1)
	{
		fprintf(stderr, "#r\n"); // ready
								 // read commands from stdin

		while (getchar() != '#');
		int command = getchar();

		// parse command

		if (command == 'e')
		{
			// exit loop and terminate
			break;
		}
		else if (command == 't')
		{
			// comamnded to trim edges
			// parse k0 k1 k2 k3 nonce

			scanf("%llu %llu %llu %llu %llu", &k0, &k1, &k2, &k3, &nonce);
			fprintf(stderr, "#a\n"); // ack
			fprintf(stderr, "Trimming: %llx %llx %llx %llx\n", k0, k1, k2, k3); // ack
		}
		else if (command == 's')
		{
			scanf("%llu %llu %llu %llu %llu", &k0, &k1, &k2, &k3, &nonce);
			for (int i = 0; i < 42; i++)
				scanf(" %llu", &(h_mydata[i]));
			cudaMemcpyToSymbol(recovery, h_mydata, 42 * 8);
			cudaDeviceSynchronize();

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "status symbol copy: %s\n", cudaGetErrorString(cudaStatus));

			// recover solution
			cudaMemset(indexesE2, 0, indexesSize);
			FluffyRecovery << < 512, 256 >> >(k0, k1, k2, k3, (ulonglong4 *)bufferA, (int *)indexesE2);
			cudaDeviceSynchronize();
			cudaMemcpy(hostA, indexesE2, 42 * 8, cudaMemcpyDeviceToHost);

			fprintf(stderr, "#s"); 
			for (int i = 0; i < 42; i++)
				fprintf(stderr, " %lu", hostA[i]);
			fprintf(stderr, "\n");

			continue;
		}
		else
			continue;

		cudaMemset(indexesE, 0, indexesSize);
		cudaMemset(indexesE2, 0, indexesSize);

		cudaDeviceSynchronize();


		FluffySeed2A << < 512, 64 >> > (k0, k1, k2, k3, (ulonglong4 *)bufferMid, (int *)indexesE2);
		FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferMid, (ulonglong4 *)bufferA, (const int *)indexesE2, (int *)indexesE, 0);
		FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferMid, (ulonglong4 *)&((char *)bufferA)[bufferSize / 2], (const int *)indexesE2, (int *)indexesE, 32);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "status memcpy: %s\n", cudaGetErrorString(cudaStatus));


		cudaMemset(indexesE2, 0, indexesSize);
		FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
		cudaMemset(indexesE, 0, indexesSize);
		FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES / 2> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
		cudaMemset(indexesE2, 0, indexesSize);
		FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 2> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
		cudaMemset(indexesE, 0, indexesSize);
		FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 2> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
		cudaMemset(indexesE2, 0, indexesSize);
		FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 4> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);


		cudaDeviceSynchronize();

		for (int i = 0; i < 80; i++)
		{
			cudaMemset(indexesE, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES / 4, DUCK_B_EDGES / 4> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
			cudaMemset(indexesE2, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES / 4, DUCK_B_EDGES / 4> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);

		}

		cudaMemset(indexesE, 0, indexesSize);
		cudaDeviceSynchronize();

		FluffyTail << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
		cudaMemcpy(hostA, indexesE, 64 * 64 * 4, cudaMemcpyDeviceToHost);

		int pos = hostA[0];
		if (pos > 0 && pos < 500000)
			cudaMemcpy(&((u64 *)buffer)[0], &((u64 *)bufferA)[0], pos * 8, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		fprintf(stderr, "Trimmed to: %d edges\n", pos);

#ifdef _WIN32
		if (sharedData != NULL)
		{
			sharedData[0] = pos;
			for (int i = 0; i < pos; i++)
			{
				sharedData[i * 2 + 0 + 1] = buffer[i * 2 + 0];
				sharedData[i * 2 + 1 + 1] = buffer[i * 2 + 1];
			}
		}
		else
		{
			fprintf(stderr, "Memory mapped file write error!\n");
			goto Error;
		}
#else
		{
			try
			{
				auto myfile = std::fstream("edges/data.bin", std::ios::out | std::ios::binary);
				myfile.write((const char *)&pos, 4);
				myfile.write((const char *)buffer, pos * 8);
				myfile.close();
			}
			catch (std::exception e)
			{
				fprintf(stderr, "Error writing edges to file %s!\n", e.what());
			}
		}
#endif
		fprintf(stderr, "#e %d \n", pos);


	}

	delete buffer;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "status: %s\n", cudaGetErrorString(cudaStatus));


Error:
#ifdef _WIN32
	if (handle != NULL)
		CloseHandle(handle);
#else
#endif

	fprintf(stderr, "CUDA terminating...\n");
	fprintf(stderr, "#x\n");
	cudaFree(bufferA);
	cudaFree(bufferB);
	cudaFree(indexesE);
	cudaFree(indexesE2);
	cudaDeviceReset();
	return 0;
}
