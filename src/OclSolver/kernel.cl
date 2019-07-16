// Cuckarood Cycle, a memory-hard proof-of-work by John Tromp and team Grin
// Copyright (c) 2018 Jiri Photon Vadura and John Tromp
// This GGM miner file is covered by the FAIR MINING license

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef uchar u8;
typedef ushort u16;
typedef uint u32;
typedef ulong u64;

typedef u32 node_t;
typedef u64 nonce_t;


#define DUCK_SIZE_A 132L
#define DUCK_SIZE_B 86L

#define DUCK_A_EDGES_4K (DUCK_SIZE_A * 1024)
#define DUCK_B_EDGES_4K (DUCK_SIZE_B * 1024)

#define DUCK_A_EDGES (DUCK_A_EDGES_4K * 4096)
#define DUCK_B_EDGES (DUCK_B_EDGES_4K * 4096)

#define EDGE_BLOCK_SIZE (64)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)

#define EDGEBITS 29
// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK (NEDGES - 1)

#define NEDGES2 ((node_t)1 << EDGEBITS)
#define NEDGES1 (NEDGES2/2)
#define NNODES1 NEDGES1
#define NNODES2 NEDGES2
#define NODE1MASK (NNODES1 - 1)

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

#define CTHREADS 1024
#define BKTMASK4K (4096-1)
#define BKTGRAN 32

#define EDGECNT 553648128
#define BUKETS 4096
#define BUKET_MASK (BUKETS-1)
#define BUKET_SIZE (EDGECNT/BUKETS)

#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = rotate(v1,(ulong)13); \
    v3 = rotate(v3,(ulong)16); v1 ^= v0; v3 ^= v2; \
    v0 = rorX(v0); v2 += v1; v0 += v3; \
    v1 = rotate(v1,(ulong)17);   v3 = rotate(v3,(ulong)25); \
    v1 ^= v2; v3 ^= v0; v2 = rorX(v2); \
  } while(0)

static inline ulong rorX(ulong vw) { uint2 v = as_uint2(vw);	return as_ulong((uint2)(v.y, v.x)); }

inline void Increase2bCounter(__local u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;
	atomic_or(ecounters + word, mask);
}

inline bool Read2bCounter(__local u32 * ecounters, const int _bucket)
{
	int bucket = _bucket ^ 1;
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;
	return (ecounters[word] & mask) > 0;
}

inline void Increase2bCounterC(__local u32 * ecounters, const int edge)
{
	u32 bucket = edge >> 12;
	int word = (bucket >> 5) << 1;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;
	atomic_or(ecounters + word, mask);
}

inline bool Read2bCounterCX(__global u16 * aux, __local u32 * ecounters, const int edge)
{
	u32 em = (edge >> 12) ^ 1;
	char bit = em & 0x1F;
	bool glb = bit >= 18;
	bit = glb ? bit - 16 : bit;
	int word = glb ? em >> 5 : (em >> 5) << 1;
	u32 mask = 1 << bit;
	u32 bf = glb ? aux[word] : ecounters[word];
	return (bf & mask) > 0;
}
inline void Increase2bCounterJ(__local u32 * ecounters, const int edge)
{
	u32 bucket = edge >> 12;
	int word = (bucket >> 5) << 1;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;
	atomic_or(ecounters + word, mask);
}
inline bool Read2bCounterJX(__global u16 * aux, __local u32 * ecounters, const int edge)
{
	u32 em = (edge >> 12) ^ 1;
	char bit = em & 0x1F;
	bool glb = bit >= 18;
	bit = glb ? bit - 16 : bit;
	int word = glb ? em >> 5 : (em >> 5) << 1;
	u32 mask = 1 << bit;
	u32 bf = glb ? aux[word] : ecounters[word];
	return (bf & mask) > 0;
}

inline void CheckAndStore1A(__global uint4 * dest, uint2 edgeA, __local u64 * magazine, __global int * destinationIndexes, const int group)
{
	const int bktInSize = BUKET_SIZE;
	const int bktOutSize = DUCK_B_EDGES_4K / 2;
	const u32 mask = 0x0003FFFF;
	const u64 notmask = ~((u64)mask);

	short bucket = edgeA.y & BUKET_MASK;
	u32 bits = (u32)magazine[bucket] & mask;

	u64 edgeB = atomic_exchange_explicit((volatile __local atomic_ulong *)&magazine[bucket], (u64)bits, memory_order_relaxed, memory_scope_work_group);
	if ((edgeB & notmask) == 0)
	{
		u64 edge64 = (((u64)edgeA.y) << (32 + 3)) | (((u64)edgeA.x << 6) & notmask) | bits;
		u64 res = atom_cmpxchg(&magazine[bucket], (u64)bits, edge64);
		if (res != (u64)bits)
		{
			int bktIdx = ((min(atomic_add(destinationIndexes + bucket, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize - 2));
			dest[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edgeA.y, edgeA.x, 0, 0);
		}
	}
	else
	{
		int bktIdx = ((min(atomic_add(destinationIndexes + bucket, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize - 2));
		dest[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edgeA.y, edgeA.x, edgeB >> (32 + 3), ((edgeB >> 6) & 0x1FFFF000) | (group));
	}
}
inline void CheckAndStore1B(__global uint4 * dest, uint2 edgeA, __local u64 * magazine, __global int * destinationIndexes, const int group)
{
	const int bktInSize = BUKET_SIZE;
	const int bktOutSize = DUCK_B_EDGES_4K / 2;
	const u32 mask = 0x0003FFFF;
	const u64 notmask = ~((u64)mask);

	short bucket = edgeA.y & BUKET_MASK;
	u32 bits = (u32)magazine[bucket] & mask;

	u64 edgeB = atomic_exchange_explicit((volatile __local atomic_ulong *)&magazine[bucket], (u64)bits, memory_order_relaxed, memory_scope_work_group);
	if ((edgeB & notmask) == 0)
	{
		u64 edge64 = (((u64)edgeA.y) << (32 + 3)) | (((u64)edgeA.x << 6) & notmask) | bits;
		u64 res = atom_cmpxchg(&magazine[bucket], (u64)bits, edge64);
		if (res != (u64)bits)
		{
			int bktIdx = ((min(atomic_add(destinationIndexes + bucket + 4096, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize - 2));
			dest[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edgeA.y, edgeA.x, 0, 0);
		}
	}
	else
	{
		int bktIdx = ((min(atomic_add(destinationIndexes + bucket + 4096, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize - 2));
		dest[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edgeA.y, edgeA.x, edgeB >> (32 + 3), ((edgeB >> 6) & 0x1FFFF000) | (group + 2048));
	}
}

inline void CheckAndStore2J(__global uint4 * dest, uint2 edgeA, __local u64 * magazine, __global int * destinationIndexes, const int group)
{
	const int bktInSize = DUCK_B_EDGES_4K / 2;
	const int bktOutSize = 65536;
	const u32 mask = 0x0003FFFF;
	const u64 notmask = ~((u64)mask);

	short bucket = (edgeA.y) & BUKET_MASK; // EXPERIMENT
	u32 bits = (u32)magazine[bucket] & mask;

	u64 edgeB = atomic_exchange_explicit((volatile __local atomic_ulong *)&magazine[bucket], (u64)bits, memory_order_relaxed, memory_scope_work_group);
	if ((edgeB & notmask) == 0)
	{
		u64 edge64 = (((u64)edgeA.y) << (32 + 3)) | (((u64)edgeA.x << 6) & notmask) | bits;
		u64 res = atom_cmpxchg(&magazine[bucket], (u64)bits, edge64);
		if (res != (u64)bits)
		{
			int bktIdx = ((min(atomic_add(destinationIndexes + bucket, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize));
			dest[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edgeA.y, edgeA.x, 0, 0);
		}
	}
	else
	{
		int bktIdx = ((min(atomic_add(destinationIndexes + bucket, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize));
		dest[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edgeA.y, edgeA.x, edgeB >> (32 + 3), ((edgeB >> 6) & 0x1FFFF000) | (group));
	}
}

#define SIPBLOCK(b) \
	{\
		v3 ^= blockNonce + b;\
		SIPROUND;SIPROUND;\
		v0 ^= blockNonce + b;\
		v2 ^= 0xff;\
		SIPROUND;SIPROUND;SIPROUND;SIPROUND;\
	}
	
static inline u32 ToNode(u32 edge)
{
	return ((edge&EDGEMASK) >> 17) | ((edge&ZMASK) << 12);
}
static inline u32 FromNode(u32 edge)
{
	return ((edge&EDGEMASK) >> 12) | ((edge&BUKET_MASK) << 17);
}
#define FDUMP(E, dir)\
{\
	u64 lookup = E;\
	uint2 edge1 = (uint2)( ToNode((lookup & NODE1MASK) << 1 | dir), ToNode((lookup >> 31) & (NODE1MASK << 1) | dir) );\
	short bucket = edge1.x & BUKET_MASK;\
	\
	u64 stEdge = atom_xchg(&magazine[bucket], (u64)0);\
	if (stEdge == 0)\
	{\
		u64 edge64 = (((u64)edge1.y) << 32) | edge1.x;\
		u64 res = atom_cmpxchg(&magazine[bucket], 0, edge64);\
		if (res != 0)\
		{\
			int position = (min((int) (BUKET_SIZE - 4), (int)((atomic_add(indexes + (bucket), 2)))) + ((int)bucket*32))%(BUKET_SIZE-4);\
			int idx = (BUKET_SIZE * (bucket%2048) + position) / 2;\
			__global uint4 * buffer = ((bucket < 2048) ? bufferA + (4096 * DUCK_B_EDGES_4K/4) : bufferB);\
			buffer[idx] = (uint4)(edge64, edge64 >> 32, 0, 0);\
		}\
	}\
	else\
	{\
		int position = ((int)min(BUKET_SIZE - 4, (int)((atomic_add(indexes + (bucket), 2)))) + ((int)bucket*32))%(BUKET_SIZE-4);\
		int idx = (BUKET_SIZE * (bucket%2048) + position) / 2;\
		__global uint4 * buffer = ((bucket < 2048) ? bufferA + (4096 * DUCK_B_EDGES_4K/4) : bufferB);\
		buffer[idx] = (uint4)(stEdge, stEdge >> 32, edge1.x, edge1.y);\
	}\
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel  void FluffySeed4K(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, __global uint4 * bufferA, __global uint4 * bufferB, __global u32 * indexes, const int offset, __global ulong4 * aux)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);
	const int group = get_group_id(0);

	__global ulong4 * sipblockL = aux + (((256 * 64 + 32)*(group % 256)) / 4);
	__local u64 magazine[4096];

	u64 v0, v1, v2, v3;

	for (short i = 0; i < 16; i++)
		magazine[lid + (256 * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = 0; i < 2048; i += EDGE_BLOCK_SIZE)
	{
		u64 blockNonce = gid * 2048 + i;

		v0 = v0i;
		v1 = v1i;
		v2 = v2i;
		v3 = v3i;

		for (short b = 0; b < 56; b += 4)
		{
			SIPBLOCK(b);
			u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
			SIPBLOCK(b + 1);
			u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
			SIPBLOCK(b + 2);
			u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
			SIPBLOCK(b + 3);
			u64 e4 = (v0 ^ v1) ^ (v2  ^ v3);
			sipblockL[(b*256/4)+lid] = (ulong4)(e1, e2, e3, e4);
		}

		SIPBLOCK(56);
		u64 e56 = (v0 ^ v1) ^ (v2  ^ v3);
		SIPBLOCK(57);
		u64 e57 = (v0 ^ v1) ^ (v2  ^ v3);
		SIPBLOCK(58);
		u64 e58 = (v0 ^ v1) ^ (v2  ^ v3);
		SIPBLOCK(59);
		u64 e59 = (v0 ^ v1) ^ (v2  ^ v3);

		SIPBLOCK(60);
		u64 e1 = (v0 ^ v1) ^ (v2  ^ v3);
		SIPBLOCK(61);
		u64 e2 = (v0 ^ v1) ^ (v2  ^ v3);
		SIPBLOCK(62);
		u64 e3 = (v0 ^ v1) ^ (v2  ^ v3);
		SIPBLOCK(63);
		u64 last = (v0 ^ v1) ^ (v2  ^ v3);

		FDUMP(last, 1);
		FDUMP(e1 ^ last, 0);
		FDUMP(e2 ^ last, 1);
		FDUMP(e3 ^ last, 0);

		FDUMP(e56 ^ last, 0);
		FDUMP(e57 ^ last, 1);
		FDUMP(e58 ^ last, 0);
		FDUMP(e59 ^ last, 1);

		for (short s = 13; s >= 0; s--)
		{
			ulong4 edges = sipblockL[s*256+lid];
			FDUMP(edges.x ^ last, 0);
			FDUMP(edges.y ^ last, 1);
			FDUMP(edges.z ^ last, 0);
			FDUMP(edges.w ^ last, 1);
		}

	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < 16; i++)
	{
		int bucket = lid + (256 * i);
		u64 edge = magazine[bucket];
		if (edge != 0)
		{
			int position = (((int)min(BUKET_SIZE - 4, (int)(atomic_add(indexes + (bucket), 2))) + ((int)bucket * 32)) % (BUKET_SIZE - 4));
			int idx = ((bucket % 2048) * BUKET_SIZE + position) / 2;
			__global uint4 * buffer = (bucket < 2048 ? bufferA + (4096 * DUCK_B_EDGES_4K / 4) : bufferB);
			buffer[idx] = (uint4)(edge, edge >> 32, 0, 0);
		}
	}

}

#define R1A_WG 1024
__attribute__((reqd_work_group_size(R1A_WG, 1, 1)))
__kernel   void FluffyRound1A(__global uint2 * buffer, const __global int * sourceIndexes, __global int * destinationIndexes, __global u16 * ax)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = BUKET_SIZE;
	const int bktOutSize = DUCK_B_EDGES_4K / 2;
	const int offset = bktInSize * group;

	__global const uint2 * source = buffer + (4096 * DUCK_B_EDGES_4K / 2);
	__global u16 * aux = ax + (group * 4096);
	__global uint4 * destination = (__global uint4 *)buffer;

	__local u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const short loops = (edgesInBucket + R1A_WG) / R1A_WG;

	for (int i = 0; i < (8192 / R1A_WG); i++)
		ecounters[lid + (R1A_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = 0; i < loops; i++)
	{
		int lindex = mad24(i, R1A_WG, lid);

		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + (group * 32)) % (BUKET_SIZE - 4));
			uint2 edge = source[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			Increase2bCounterC(ecounters, edge.x);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	u64 a = ((u64*)(ecounters))[lid * 4 + 0];
	u64 b = ((u64*)(ecounters))[lid * 4 + 1];
	u64 c = ((u64*)(ecounters))[lid * 4 + 2];
	u64 d = ((u64*)(ecounters))[lid * 4 + 3];
	((ushort4*)(aux))[lid] = (ushort4)(a >> 16, b >> 16, c >> 16, d >> 16);
	((u64*)(ecounters))[lid * 4 + 0] = a & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 1] = b & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 2] = c & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 3] = d & 0x000000000003FFFF;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loops - 1; i >= 0; i--)
	{
		int lindex = mad24(i, R1A_WG, lid);

		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + (group * 32)) % (BUKET_SIZE - 4));
			uint2 edge = source[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			bool lives = Read2bCounterCX(aux, ecounters, edge.x);
			if (lives)
			{
				CheckAndStore1A(destination, edge, (__local u64 *)ecounters, destinationIndexes, group);
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	const u32 mask = 0x0003FFFF;
	const u64 notmask = ~((u64)mask);
	for (int i = 0; i < 4; i++)
	{
		int bucket = lid + (1024 * i);
		u64 edge = ((__local u64 *)ecounters)[bucket];
		if ((edge & notmask) != 0)
		{
			const int bucket = (edge >> (32 + 3)) & BKTMASK4K;
			const int bktIdx = (min(atomic_add(destinationIndexes + bucket, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize - 2);
			destination[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edge >> (32 + 3), ((edge >> 6) & 0x1FFFF000) | (group), 0, 0);
		}
	}
}

#define R1B_WG 1024
__attribute__((reqd_work_group_size(R1B_WG, 1, 1)))
__kernel   void FluffyRound1B(const __global uint2 * src, __global uint4 * dst, const __global int * sourceIndexes, __global int * destinationIndexes, __global u16 * ax)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = BUKET_SIZE;
	const int bktOutSize = DUCK_B_EDGES_4K / 2;
	const int offset = bktInSize * group;

	__global uint4 * destination = dst + (4096 * DUCK_B_EDGES_4K / 4);
	__global u16 * aux = ax + (group * 4096);

	__local u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group + 2048], bktInSize);
	const short loops = (edgesInBucket + R1B_WG) / R1B_WG;

	for (int i = 0; i < (8192 / R1B_WG); i++)
		ecounters[lid + (R1B_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = 0; i < loops; i++)
	{
		int lindex = mad24(i, R1B_WG, lid);

		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group + 2048) * 32)) % (BUKET_SIZE - 4));
			uint2 edge = src[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			Increase2bCounterC(ecounters, edge.x);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	u64 a = ((u64*)(ecounters))[lid * 4 + 0];
	u64 b = ((u64*)(ecounters))[lid * 4 + 1];
	u64 c = ((u64*)(ecounters))[lid * 4 + 2];
	u64 d = ((u64*)(ecounters))[lid * 4 + 3];
	((ushort4*)(aux))[lid] = (ushort4)(a >> 16, b >> 16, c >> 16, d >> 16);
	((u64*)(ecounters))[lid * 4 + 0] = a & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 1] = b & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 2] = c & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 3] = d & 0x000000000003FFFF;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loops - 1; i >= 0; i--)
	{
		int lindex = mad24(i, R1B_WG, lid);

		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group + 2048) * 32)) % (BUKET_SIZE - 4));
			uint2 edge = src[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			bool lives = Read2bCounterCX(aux, ecounters, edge.x);
			if (lives)
			{
				CheckAndStore1B(destination, edge, (__local u64 *)ecounters, destinationIndexes, group);
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	const u32 mask = 0x0003FFFF;
	const u64 notmask = ~((u64)mask);
	for (int i = 0; i < 4; i++)
	{
		int bucket = lid + (1024 * i);
		u64 edge = ((__local u64 *)ecounters)[bucket];
		if ((edge & notmask) != 0)
		{
			const int bucket = (edge >> (32 + 3)) & BKTMASK4K;
			const int bktIdx = (min(atomic_add(destinationIndexes + bucket + 4096, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize - 2);
			destination[((bucket * bktOutSize) + bktIdx) / 2] = (uint4)(edge >> (32 + 3), ((edge >> 6) & 0x1FFFF000) | (group + 2048), 0, 0);
		}
	}
}

#define R2J_WG 1024
__attribute__((reqd_work_group_size(R2J_WG, 1, 1)))
__kernel   void FluffyRound2J(const __global uint2 * srcA, __global uint2 * dst, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = DUCK_B_EDGES_4K / 2;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;
	const __global uint2 * srcB = srcA + (4096 * DUCK_B_EDGES_4K / 2);

	__local u32 ecounters[8192];

	const int edgesInBucketA = min(sourceIndexes[group], bktInSize);
	const short loopsA = (edgesInBucketA + R2J_WG) / R2J_WG;
	const int edgesInBucketB = min(sourceIndexes[group + 4096], bktInSize);
	const short loopsB = (edgesInBucketB + R2J_WG) / R2J_WG;

	for (int i = 0; i < (8192 / R2J_WG); i++)
		ecounters[lid + (R2J_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = 0; i < loopsA; i++)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketA)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcA[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}
	for (short i = 0; i < loopsB; i++)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketB)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcB[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loopsB - 1; i >= 0; i--)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketB)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcB[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = (edge.y) & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}
	for (short i = loopsA - 1; i >= 0; i--)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketA)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcA[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = (edge.y) & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

}

__attribute__((reqd_work_group_size(R2J_WG, 1, 1)))
__kernel   void FluffyRound2JX(const __global uint2 * srcA, __global uint4 * dst, const __global int * sourceIndexes, __global int * destinationIndexes, __global u16 * ax)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = DUCK_B_EDGES_4K / 2;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;
	const __global uint2 * srcB = srcA + (4096 * DUCK_B_EDGES_4K / 2);
	__global u16 * aux = ax + (group * 4096);

	__local u32 ecounters[8192];

	const int edgesInBucketA = min(sourceIndexes[group], bktInSize);
	const short loopsA = (edgesInBucketA + R2J_WG) / R2J_WG;
	const int edgesInBucketB = min(sourceIndexes[group + 4096], bktInSize);
	const short loopsB = (edgesInBucketB + R2J_WG) / R2J_WG;

	for (int i = 0; i < (8192 / R2J_WG); i++)
		ecounters[lid + (R2J_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = 0; i < loopsA; i++)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketA)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcA[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			Increase2bCounterJ(ecounters, edge.x);
		}
	}
	for (short i = 0; i < loopsB; i++)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketB)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcB[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			Increase2bCounterJ(ecounters, edge.x);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	u64 a = ((u64*)(ecounters))[lid * 4 + 0];
	u64 b = ((u64*)(ecounters))[lid * 4 + 1];
	u64 c = ((u64*)(ecounters))[lid * 4 + 2];
	u64 d = ((u64*)(ecounters))[lid * 4 + 3];
	((ushort4*)(aux))[lid] = (ushort4)(a >> 16, b >> 16, c >> 16, d >> 16);
	((u64*)(ecounters))[lid * 4 + 0] = a & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 1] = b & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 2] = c & 0x000000000003FFFF;
	((u64*)(ecounters))[lid * 4 + 3] = d & 0x000000000003FFFF;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loopsB - 1; i >= 0; i--)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketB)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcB[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			bool lives = Read2bCounterJX(aux, ecounters, edge.x);
			if (lives)
				CheckAndStore2J(dst, edge, (__local u64 *)ecounters, destinationIndexes, group);
		}
	}
	for (short i = loopsA - 1; i >= 0; i--)
	{
		int lindex = mad24(i, R2J_WG, lid);
		if (lindex < edgesInBucketA)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize - 2));
			uint2 edge = srcA[index];
			bool bypass = edge.x == 0 && edge.y == 0;
			if (bypass) continue;
			bool lives = Read2bCounterJX(aux, ecounters, edge.x);
			if (lives)
				CheckAndStore2J(dst, edge, (__local u64 *)ecounters, destinationIndexes, group);
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	const u32 mask = 0x0003FFFF;
	const u64 notmask = ~((u64)mask);
	for (int i = 0; i < 4; i++)
	{
		int bucket = lid + (1024 * i);
		u64 edge = ((__local u64 *)ecounters)[bucket];
		if ((edge & notmask) != 0)
		{
			const int bucket = (edge >> (32 + 3)) & BKTMASK4K;
			const int bktIdx = (min(atomic_add(destinationIndexes + bucket, 2), bktOutSize - 2) + (bucket * 32)) % (bktOutSize);
		}
	}
}

#define R310_WG 1024
__attribute__((reqd_work_group_size(R310_WG, 1, 1)))
__kernel   void FluffyRound3_5(const __global uint2 * src, __global uint2 * dst, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = 65536;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;

	__local u32 ecounters[8192];
	uint2 rbuf[7];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const short loops = (edgesInBucket + R310_WG) / R310_WG;

	for (int i = 0; i < (8192 / R310_WG); i++)
		ecounters[lid + (R310_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	#pragma unroll 7
	for (short i = 0; i < 7; i++)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			rbuf[i] = src[index];
			Increase2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12);
		}
	}

	for (short i = 7; i < loops; i++)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loops - 1; i >= 7; i--)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

	#pragma unroll 7
	for (short i = 0; i < 7; i++)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			if (Read2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12))
			{
				const int bucket = rbuf[i].y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(rbuf[i].y, rbuf[i].x);
			}
		}
	}

}

__attribute__((reqd_work_group_size(R310_WG, 1, 1)))
__kernel   void FluffyRound6_10(const __global uint2 * src, __global uint2 * dst, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = 65536;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;

	__local u32 ecounters[8192];
	uint2 rbuf[4];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const short loops = (edgesInBucket + R310_WG) / R310_WG;

	for (int i = 0; i < (8192 / R310_WG); i++)
		ecounters[lid + (R310_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll 4
	for (short i = 0; i < 4; i++)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			rbuf[i] = src[index];
			Increase2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12);
		}
	}

	for (short i = 4; i < loops; i++)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loops - 1; i >= 4; i--)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

#pragma unroll 4
	for (short i = 0; i < 4; i++)
	{
		int lindex = mad24(i, R310_WG, lid);
		if (lindex < edgesInBucket)
		{
			if (Read2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12))
			{
				const int bucket = rbuf[i].y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(rbuf[i].y, rbuf[i].x);
			}
		}
	}

}

#define R11_WG 256
__attribute__((reqd_work_group_size(R11_WG, 1, 1)))
__kernel   void FluffyRound11(const __global uint2 * src, __global uint2 * dst, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = 65536;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;

	__local u32 ecounters[8192];
	uint2 rbuf[8];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const short loops = (edgesInBucket + R11_WG) / R11_WG;

	for (int i = 0; i < (8192 / R11_WG); i++)
		ecounters[lid + (R11_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll 8
	for (short i = 0; i < 8; i++)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			rbuf[i] = src[index];
			Increase2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12);
		}
	}

	for (short i = 8; i < loops; i++)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loops - 1; i >= 8; i--)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

#pragma unroll 8
	for (short i = 0; i < 8; i++)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			if (Read2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12))
			{
				const int bucket = rbuf[i].y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(rbuf[i].y, rbuf[i].x);
			}
		}
	}

}

#define R11_WG 256
__attribute__((reqd_work_group_size(R11_WG, 1, 1)))
__kernel   void FluffyRound15(const __global uint2 * src, __global uint2 * dst, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = 65536;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;

	__local u32 ecounters[8192];
	uint2 rbuf[4];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const short loops = (edgesInBucket + R11_WG) / R11_WG;

	for (int i = 0; i < (8192 / R11_WG); i++)
		ecounters[lid + (R11_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll 4
	for (short i = 0; i < 4; i++)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			rbuf[i] = src[index];
			Increase2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12);
		}
	}

	for (short i = 4; i < loops; i++)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loops - 1; i >= 4; i--)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

#pragma unroll 4
	for (short i = 0; i < 4; i++)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			if (Read2bCounter(ecounters, (rbuf[i].x & EDGEMASK) >> 12))
			{
				const int bucket = rbuf[i].y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(rbuf[i].y, rbuf[i].x);
			}
		}
	}

}

#define R11_WG 256
__attribute__((reqd_work_group_size(R11_WG, 1, 1)))
__kernel   void FluffyRound23(const __global uint2 * src, __global uint2 * dst, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = 65536;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;

	__local u32 ecounters[8192];
	uint2 rbuf;

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const short loops = (edgesInBucket + R11_WG) / R11_WG;

	for (int i = 0; i < (8192 / R11_WG); i++)
		ecounters[lid + (R11_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	{
		int lindex = lid;
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			rbuf = src[index];
			Increase2bCounter(ecounters, (rbuf.x & EDGEMASK) >> 12);
		}
	}

	for (short i = 1; i < loops; i++)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = loops - 1; i >= 1; i--)
	{
		int lindex = mad24(i, R11_WG, lid);
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			uint2 edge = src[index];
			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

	{
		int lindex = lid;
		if (lindex < edgesInBucket)
		{
			if (Read2bCounter(ecounters, (rbuf.x & EDGEMASK) >> 12))
			{
				const int bucket = rbuf.y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(rbuf.y, rbuf.x);
			}
		}
	}

}

#define R11_WG 256
__attribute__((reqd_work_group_size(R11_WG, 1, 1)))
__kernel   void FluffyRound80(const __global uint2 * src, __global uint2 * dst, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);
	const int bktInSize = 65536;
	const int bktOutSize = 65536;
	const int offset = bktInSize * group;

	__local u32 ecounters[8192];
	uint2 rbuf;

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);

	for (int i = 0; i < (8192 / R11_WG); i++)
		ecounters[lid + (R11_WG * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	{
		int lindex = lid;
		if (lindex < edgesInBucket)
		{
			const int index = offset + ((lindex + ((group) * 32)) % (bktInSize));
			rbuf = src[index];
			Increase2bCounter(ecounters, (rbuf.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	{
		int lindex = lid;
		if (lindex < edgesInBucket)
		{
			if (Read2bCounter(ecounters, (rbuf.x & EDGEMASK) >> 12))
			{
				const int bucket = rbuf.y & BKTMASK4K;
				const int bktIdx = (min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1) + (bucket * 32)) % (bktOutSize);
				dst[(bucket * bktOutSize) + bktIdx] = (uint2)(rbuf.y, rbuf.x);
			}
		}
	}

}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel   void FluffyRecovery(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const __constant u64 * recovery, __global int * indexes)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);

	__local u32 nonces[42];
	u64 sipblock[64];

	u64 v0;
	u64 v1;
	u64 v2;
	u64 v3;

	if (lid < 42) nonces[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

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
				bool match = (recovery[i] == a) || (recovery[i] == b);
				if (match)
					nonces[i] = blockNonce + s;
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < 42)
	{
		if (nonces[lid] > 0)
			indexes[lid] = nonces[lid];
	}
}

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void FluffyTailO(const __global uint2 * source, __global uint2 * destination, const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	int myEdges = sourceIndexes[group];
	__local int destIdx;

	if (lid == 0)
		destIdx = atomic_add(destinationIndexes, myEdges);

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < myEdges)
	{
		uint2 src = source[(group * 65536) + ((lid + (group * 32)) % 65536)];
		destination[destIdx + lid] = (uint2)( FromNode(src.x), FromNode(src.y) );
	}
}
