typedef uint8 u8;
typedef uint16 u16;
typedef uint u32;
typedef ulong u64;
typedef u32 node_t;
typedef u64 nonce_t;

#define DEBUG 1

// this define should be set from host code to be completely universal
#define EDGEBITS 29
// number of edges
#define NEDGES ((u64)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK (NEDGES - 1)

#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = rotate(v1,(ulong)13); \
    v3 = rotate(v3,(ulong)16); v1 ^= v0; v3 ^= v2; \
    v0 = rotate(v0,(ulong)32); v2 += v1; v0 += v3; \
    v1 = rotate(v1,(ulong)17);   v3 = rotate(v3,(ulong)21); \
    v1 ^= v2; v3 ^= v0; v2 = rotate(v2,(ulong)32); \
  } while(0)

u64 dipnode(ulong v0i, ulong v1i, ulong v2i, ulong v3i, u64 nce, uint uorv) {
	ulong nonce = 2 * nce + uorv;
	ulong v0 = v0i, v1 = v1i, v2 = v2i, v3 = v3i ^ nonce;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

#define MODE_SETCNT 1
#define MODE_TRIM 2
#define MODE_EXTRACT 3

// Minimalistic cuckatoo lean trimmer
// This implementation is not optimal!
//
// 8 global kernel executions (hardcoded ATM)
// 1024 thread blocks, 256 threads each, 256 edges for each thread
// 8*1024*256*256 = 536 870 912 edges = cuckatoo29
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel  void LeanRound(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, __global uint8 * edges, __global uint * counters, __global int * aux, const int mode, const int uorv)
{
	const int blocks = NEDGES / 32;
	const int gid = get_global_id(0);
	{
		u32 el[8];
		int lCount = 0;
		// what 256 nit block of edges are we processing
		u64 index = gid;
		u64 start = index * 256;
		// load all 256 bits (edges) to registers
		uint8 load = edges[index];
		// map to an array for easier indexing (depends on compiler/GPU, could be pushed out to cache)
		el[0] = load.s0;
		el[1] = load.s1;
		el[2] = load.s2;
		el[3] = load.s3;
		el[4] = load.s4;
		el[5] = load.s5;
		el[6] = load.s6;
		el[7] = load.s7;
		
		// process as 8 x 32bit segment, GPUs have 32bit ALUs 
		for (short i = 0; i < 8; i++)
		{
			// shortcut to current 32bit value
			uint ee = el[i];
			// how many edges we process in the block
			short lEdges = popcount(ee);
			// whole warp will always execute worst case scenario, but it will help in the long run (not benched)
			
			// now a loop for every single living edge in current 32 edge block
			for (short e = 0; e < lEdges; e++)
			{
				// bit position of next living edge
				short pos = ctz(ee);
				// position in the 256 edge block
				int subPos = (i * 32) + pos;
				// reconstruct value of noce for this edge
				int nonce = start + subPos;
				// calculate siphash24 for either U or V (host device control)
				u32 hash = dipnode(v0i, v1i, v2i, v3i, nonce, uorv);
				
				// this time we set edge bit counters - PASS 1
				if (mode == MODE_SETCNT)
				{
					// what global memory 32bit block we need to access
					int block = hash / 32;
					// what bit in the block we need to set
					u32 bit = hash % 32;
					// create a bitmask from that bit
					u32 mask = (u32)1 << bit;
					// global atomic or (set bit to 1 no matter what it was)
					atomic_or(&counters[block], mask);
				}
				// this time counters are already set so need to figure out if the edge lives - PASS 2
				else if ((mode == MODE_TRIM) || (mode == MODE_EXTRACT))
				{
					// cuckatoo XOR thing
					hash = hash ^ 1;
					// what global memory 32bit block we need to read
					int block = hash / 32;
					// what bit in the block we need to read
					u32 bit = hash % 32;
					// create a bitmask from that bit
					u32 mask = (u32)1 << bit;
					// does the edge live or not
					bool lives = ((counters[block]) & mask) > 0;
					// if edge is not alive, kill it (locally in registers)
					if (!lives)
					{
						// TODO: is this correct bit?
						el[i] ^= (u32)1 << pos; // 1 XOR 1 is 0
					}
					else
					{
						// debug counter of alive edges
						if (DEBUG)
							lCount++;

						// if this is last lean round we do, store all edges in one long list
						if (mode == MODE_EXTRACT) // PASS N_rounds
						{
							// obtain global pointer to final edge list
							int edgePos = atomic_inc(aux+1);
							// position in output array as multiple of 128bits (32bits will be empty)
							int auxIndex = 4 + (edgePos * 4);
							// debug failsafe
							if (!(DEBUG && (auxIndex > (1024 * 1024))))
							{
								// store all information to global memory
								aux[auxIndex + 0] = dipnode(v0i, v1i, v2i, v3i, nonce, 0);
								aux[auxIndex + 1] = dipnode(v0i, v1i, v2i, v3i, nonce, 1);
								aux[auxIndex + 2] = nonce;
								aux[auxIndex + 3] = 0; // for clarity, competely useless operation
							}
						}
					}
				}
				// clear current edge position so that we can skip it in next run of ctz()
				ee ^= (u32)1 << pos; // 1 XOR 1 is 0
			}
		}
		// return edge bits back to global memory if we are in second stage
		if (mode == MODE_TRIM)
			edges[index] = (uint8)(el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7]);
		// debug only, use aux buffer to count alive edges in this round
		if (DEBUG)
			atomic_add(aux, lCount);
	}
}
