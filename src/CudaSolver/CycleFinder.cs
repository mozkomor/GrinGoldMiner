using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharedSerialization;

namespace CudaSolver
{
    // inspided by Gringoo miner by eth_saver
    public static class FinderBag
    {
        private static ConcurrentBag<CGraph> finders = new ConcurrentBag<CGraph>();

        public static CGraph GetFinder()
        {
            if (finders.TryTake(out CGraph finder))
                return finder;
            else
            {
                return new CGraph();
            }                
        }

        public static void ReturnFinder(CGraph finder)
        {
            if (finders.Count < 10)
                finders.Add(finder);
        }
    }


    public class CGraphLegacy
    {
        public static bool ShowCycles = false;

        private Dictionary<uint, uint> graphU;
        private Dictionary<uint, uint> graphV;

        private int[] edges;
        private int maxlength = 8192;
        private int edgeCount = 0;

        private Job job;

        public int dupes = 0;

        public volatile bool SolutionFound = false;

        public CGraphLegacy()
        {
            graphU = new Dictionary<uint, uint>(edgeCount);
            graphV = new Dictionary<uint, uint>(edgeCount);
            edges = new int[200000];
        }
        
        public void SetEdges(int[] edgesExternal, int count)
        {
            edgeCount = count;
            Array.Copy(edgesExternal, edges, count * 2);
            graphU.Clear();
            graphV.Clear();
            SolutionFound = false;
            dupes = 0;
        }

        public void SetHeader(Job jobToSolve)
        {
            job = jobToSolve;
        }

        internal void FindSolutions(ConcurrentQueue<Solution> solutions, int cyclen = 42)
        {
            for (int ee = 0; ee < edgeCount; ee++)
            {
                Edge e = new Edge() { Item1 = (uint)edges[ee * 2 + 0], Item2 = (uint)edges[ee * 2 + 1] };
                {
                    if (graphU.TryGetValue(e.Item1, out uint I1) && I1 == e.Item2)
                    //if (graphU.ContainsKey(e.Item1) && graphU[e.Item1] == e.Item2)
                    {
                        dupes++;
                        continue;
                    }

                    if (graphV.TryGetValue(e.Item2, out uint I2) && I2 == e.Item1)
                    //if (graphV.ContainsKey(e.Item2) && graphV[e.Item2] == e.Item1)
                    {
                        dupes++;
                        continue;
                    }

                    {
                        List<uint> path1 = path(true, e.Item1);
                        List<uint> path2 = path(false, e.Item2);

                        long joinA = -1;
                        long joinB = -1;

                        for (int i = 0; i < path1.Count; i++)
                        {
                            uint ival = path1[i];
                            if (path2.Contains(ival))
                            {
                                var path2Idx = path2.IndexOf(ival);

                                joinA = i;
                                joinB = path2Idx;

                                break;
                            }
                        }

                        long cycle = joinA != -1 ? 1 + joinA + joinB : 0;

                        if (cycle >= 4 && cycle != cyclen)
                        {
                            if (ShowCycles)
                            {
                                Console.ForegroundColor = ConsoleColor.Yellow;
                                Console.WriteLine(cycle.ToString() + "-cycle found");
                                Console.ResetColor();
                            }
                        }
                        else if (cycle == cyclen)
                        {
                            if (ShowCycles)
                            {
                                Console.ForegroundColor = ConsoleColor.Red;
                                Console.WriteLine("42-cycle found!");
                                // initiate nonce recovery procedure
                                Console.ResetColor();
                            }

                            List<uint> path1t = path1.Take((int)joinA + 1).ToList();
                            List<uint> path2t = path2.Take((int)joinB + 1).ToList();
                            List<Edge> cycleEdges = new List<Edge>(42);
                            cycleEdges.Add(e);

                            cycleEdges.AddRange(path1t.Zip(path1t.Skip(1), (second, first) => new Edge(first, second)));
                            cycleEdges.AddRange(path2t.Zip(path2t.Skip(1), (second, first) => new Edge(first, second)));

                            SolutionFound = true;

                            lock (solutions)
                            {
                                solutions.Enqueue(new Solution() { job = this.job, edges = cycleEdges });
                            }
                        }
                        else
                        {
                            if (path1.Count > path2.Count)
                            {
                                Reverse(path2, false);
                                graphV[e.Item2] = e.Item1;
                            }
                            else
                            {
                                Reverse(path1, true);
                                graphU[e.Item1] = e.Item2;
                            }
                        }


                    }

                }

            }

        }


        private void Reverse(List<uint> path, bool startsInU)
        {
            for (int i = path.Count - 2; i >= 0; i--)
            {
                uint A = path[i];
                uint B = path[i + 1];

                if (startsInU)
                {
                    if ((i & 1) == 0)
                    {
                        graphU.Remove(A);
                        graphV[B] = A;
                    }
                    else
                    {
                        graphV.Remove(A);
                        graphU[B] = A;
                    }
                }
                else
                {
                    if ((i & 1) == 0)
                    {
                        graphV.Remove(A);
                        graphU[B] = A;
                    }
                    else
                    {
                        graphU.Remove(A);
                        graphV[B] = A;
                    }
                }
            }
        }

        internal List<uint> path(bool _startInGraphU, uint _key)
        {
            List<uint> path = new List<uint>();
            uint key = _key;
            bool startInGraphU = _startInGraphU;

            Dictionary<uint, uint> graph = _startInGraphU ? graphU : graphV;

            graph = _startInGraphU ? graphU : graphV;

            path.Add(key);

            uint v = 0;
            //while (graph.ContainsKey(key))
            while (graph.TryGetValue(key, out v))
            {
                if ((path.Count >= maxlength)) break;

                path.Add(v);

                startInGraphU = !startInGraphU;
                graph = startInGraphU ? graphU : graphV;

                key = v;
            }

            return path;
        }


    }

    /// <summary>
    /// Copy of cycle finder from John Tromp
    /// https://github.com/tromp/cuckoo/blob/master/src/cuckaroo
    /// </summary>
    public class CGraph
    {
        const int MAXSOLS = 4;
        const int IDXSHIFT = 12;
        const uint NEDGES = 1 << 29;
        const uint MAXEDGES = NEDGES >> IDXSHIFT;

        public static bool ShowCycles
        {
            set
            {
                graph.ShowCycles = value;
            }
        }

        private int[] edges;
        private int maxlength = 8192;

        private Job job;
        private graph cg;

        public int dupes = 0;

        public CGraph()
        {
            cg = new graph(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT);
        }

        public void SetEdges(int[] edgesExternal, int count)
        {
            edges = new int[count * 2];
            Array.Copy(edgesExternal, edges, count * 2);
        }

        public void SetHeader(Job jobToSolve)
        {
            job = jobToSolve;
        }

        internal void FindSolutions(ConcurrentQueue<Solution> solutions, int cyclen = 42)
        {
            cg.reset();

            for (int ee = 0; ee < edges.Length / 2; ee++)
            {
                cg.add_compress_edge((uint)edges[ee * 2 + 0], (uint)edges[ee * 2 + 1]);
            }
            for (int s = 0; s < cg.nsols; s++)
            {
                List<Edge> lst = new List<Edge>(42);
                // print_log("Solution");
                for (int j = 0; j < 42; j++)
                {
                    lst.Add(new Edge((uint)edges[cg.sols[s][j] * 2], (uint)edges[cg.sols[s][j] * 2 + 1]));
                    // print_log(" (%x, %x)", soledges[j].x, soledges[j].y);
                }

                solutions.Enqueue(new Solution() { job = this.job, edges = lst });
            }
        }


    }

    /// <summary>
    /// https://github.com/tromp/cuckoo/blob/master/src/cuckaroo
    /// FAIR MINING License
    /// </summary>
    unsafe public class graph
    {
        public static bool ShowCycles = false;

        const uint NIL = ~(uint)0;
        const uint EDGEBITS = 29;
        const uint PROOFSIZE = 42;

        public struct link
        { // element of adjacency list
            public uint next;
            public uint to;
        };

        public uint MAXEDGES;
        public uint MAXNODES;
        public uint nlinks; // aka halfedges, twice number of edges
        public uint[] adjlist; // index into links array
        public link[] links;
        public bool sharedmem;
        public compressor compressu;
        public compressor compressv;
        public bitmap visited;
        public uint MAXSOLS;
        public List<uint[]> sols;
        public uint nsols;

        public graph(uint maxedges, uint maxnodes, uint maxsols) //:  visited(2*maxnodes)
        {
            MAXEDGES = maxedges;
            MAXNODES = maxnodes;
            MAXSOLS = maxsols;
            adjlist = new uint[2 * MAXNODES]; // index into links array
            links = new link[2 * MAXEDGES];
            compressu = compressv = null;
            sharedmem = false;
            sols = new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols[i] = new uint[PROOFSIZE];
            visited = new bitmap(2 * maxnodes);
            visited.clear();
        }

        public graph(uint maxedges, uint maxnodes, uint maxsols, uint compressbits)// : visited(2*maxnodes)
        {
            MAXEDGES = maxedges;
            MAXNODES = maxnodes;
            MAXSOLS = maxsols;
            adjlist = new uint[2 * MAXNODES]; // index into links array
            links = new link[2 * MAXEDGES];
            compressu = new compressor(EDGEBITS, compressbits);
            compressv = new compressor(EDGEBITS, compressbits);
            sharedmem = false;
            sols = new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols.Add(new uint[PROOFSIZE]);
            visited = new bitmap(2 * maxnodes);
            visited.clear();
        }

        public graph(uint maxedges, uint maxnodes, uint maxsols, char* bytes)// : visited(2*maxnodes)
        {
            MAXEDGES = maxedges;
            MAXNODES = maxnodes;
            MAXSOLS = maxsols;
            adjlist = new uint[2 * MAXNODES]; // index into links array
            links = new link[2 * MAXEDGES];
            compressu = compressv = null;
            sharedmem = true;
            sols = new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols.Add(new uint[PROOFSIZE]);
            visited = new bitmap(2 * maxnodes);
            visited.clear();
        }

        public graph(uint maxedges, uint maxnodes, uint maxsols, uint compressbits, char* bytes)// : visited(2*maxnodes)
        {
            MAXEDGES = maxedges;
            MAXNODES = maxnodes;
            MAXSOLS = maxsols;
            adjlist = new uint[2 * MAXNODES]; // index into links array
            links = new link[2 * MAXEDGES];
            compressu = new compressor(EDGEBITS, compressbits);
            compressv = new compressor(EDGEBITS, compressbits);
            sharedmem = true;
            sols = new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols.Add(new uint[PROOFSIZE]);
            visited = new bitmap(2 * maxnodes);
            visited.clear();
        }

        // total size of new-operated data, excludes sols and visited bitmap of MAXEDGES bits
        public UInt64 bytes()
        {
            return 0;// sizeof(uint[2 * MAXNODES]) + sizeof(link[2 * MAXEDGES]) + (compressu ? 2 * compressu->bytes() : 0);
        }

        public void reset()
        {
            //memset(adjlist, (char)NIL, sizeof(uint[2 * MAXNODES]));
            Array.Fill(adjlist, NIL);
            if (compressu != null)
            {
                compressu.reset();
                compressv.reset();
            }
            resetcounts();
        }

        void resetcounts()
        {
            nlinks = nsols = 0;
            // visited has entries set only during cycles() call
        }

        public static unsafe bool nonce_cmp(ref uint a, ref uint b)
        {
            return a == b;
        }

        public void cycles_with_link(uint len, uint u, uint dest)
        {
            unchecked
            {
                // printf("cycles_with_link(%d, %x, %x)\n", len, u, dest);
                if (visited.test(u))
                    return;
                if (u == dest)
                {
                    if (ShowCycles)
                    {
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.WriteLine($"  {len}-cycle found");
                        Console.ResetColor();
                    }
                    if (len == PROOFSIZE && nsols < MAXSOLS)
                    {
                        //qsort(sols[nsols++], PROOFSIZE, sizeof(uint), nonce_cmp);
                        sols[(int)nsols] = sols[(int)nsols++].OrderBy(e => e).ToArray();
                        //memcpy(sols[nsols], sols[nsols - 1], sizeof(sols[0]));
                        Array.Copy(sols[(int)nsols - 1], sols[(int)nsols], sols[(int)nsols].Length);
                    }
                    return;
                }
                if (len == PROOFSIZE)
                    return;
                uint au1 = adjlist[u];
                if (au1 != NIL)
                {
                    visited.set(u);
                    for (; au1 != NIL; au1 = links[au1].next)
                    {
                        sols[(int)nsols][len] = au1 / 2;
                        cycles_with_link(len + 1, links[au1 ^ 1].to, dest);
                    }
                    visited.reset(u);
                }
            }
        }

        public void add_edge(uint u, uint v)
        {
            unchecked
            {
                //assert(u < MAXNODES);
                //assert(v < MAXNODES);
                v += MAXNODES; // distinguish partitions
                if (adjlist[u] != NIL && adjlist[v] != NIL)
                { // possibly part of a cycle
                    sols[(int)nsols][0] = nlinks / 2;
                    //assert(!visited.test(u));
                    cycles_with_link(1, u, v);
                }
                uint ulink = nlinks++;
                uint vlink = nlinks++; // the two halfedges of an edge differ only in last bit
                //assert(vlink != NIL);    // avoid confusing links with NIL; guaranteed if bits in uint > EDGEBITS + 1
                links[ulink].next = adjlist[u];
                links[vlink].next = adjlist[v];
                links[adjlist[u] = ulink].to = u;
                links[adjlist[v] = vlink].to = v;
            }
        }

        public void add_compress_edge(uint u, uint v)
        {
            add_edge(compressu.compress(u), compressv.compress(v));
        }
    };

    /// <summary>
    /// https://github.com/tromp/cuckoo/blob/master/src/cuckaroo
    /// FAIR MINING License
    /// </summary>
    unsafe public class compressor
    {
        public uint NODEBITS;
        public uint SHIFTBITS;
        public uint SIZEBITS;
        public uint SIZE;
        public uint SIZE2;
        public uint MASK;
        public uint MASK2;
        public uint nnodes;
        public const uint NIL = ~(uint)0;
        public uint[] nodes;
        public bool sharedmem;

        public compressor(uint nodebits, uint compressbits, char* bytes)
        {
            NODEBITS = nodebits;
            SHIFTBITS = compressbits;
            SIZEBITS = NODEBITS - compressbits;
            SIZE = (uint)1 << (int)SIZEBITS;
            SIZE2 = (uint)2 << (int)SIZEBITS;
            nodes = new uint[SIZE2];
            sharedmem = true;
            MASK = SIZE - 1;
            MASK2 = SIZE2 - 1;
        }

        public compressor(uint nodebits, uint compressbits)
        {
            NODEBITS = nodebits;
            SHIFTBITS = compressbits;
            SIZEBITS = NODEBITS - compressbits;
            SIZE = (uint)1 << (int)SIZEBITS;
            SIZE2 = (uint)2 << (int)SIZEBITS;
            nodes = new uint[SIZE2];
            sharedmem = false;
            MASK = SIZE - 1;
            MASK2 = SIZE2 - 1;
        }

        ~compressor()
        {
            if (!sharedmem)
                nodes = null;
        }

        public UInt64 bytes()
        {
            return sizeof(uint) * SIZE2;
        }

        public void reset()
        {
            //memset(nodes, (char)NIL, sizeof(uint[SIZE2]));
            Array.Fill(nodes, NIL);
            nnodes = 0;
        }

        public uint compress(uint u)
        {
            unchecked
            {
                uint ui = u >> (int)SHIFTBITS;
                for (; ; ui = (ui + 1) & MASK2)
                {
                    uint cu = nodes[ui];
                    if (cu == NIL)
                    {
                        if (nnodes >= SIZE)
                        {
                            //print_log("NODE OVERFLOW at %x\n", u);
                            return 0;
                        }
                        nodes[ui] = u << (int)SIZEBITS | nnodes;
                        return nnodes++;
                    }
                    if ((cu & ~MASK) == u << (int)SIZEBITS)
                    {
                        return cu & MASK;
                    }
                }
            }
        }
    };

    /// <summary>
    /// https://github.com/tromp/cuckoo/blob/master/src/cuckaroo
    /// FAIR MINING License
    /// </summary>
    unsafe public class bitmap
    {
        public uint SIZE;
        public uint BITMAP_WORDS;
        public uint[] bits;
        const uint BITS_PER_WORD = sizeof(uint) * 8;

        public bitmap(uint size)
        {
            SIZE = size;
            BITMAP_WORDS = SIZE / BITS_PER_WORD;
            bits = new uint[BITMAP_WORDS];
        }
        ~bitmap()
        {
            freebits();
        }
        void freebits()
        {
            bits = null;
        }
        public void clear()
        {
            Array.Clear(bits, 0, bits.Length);
        }
        public void prefetch(uint u)
        {

        }

        unsafe public void set(uint u)
        {
            unchecked
            {
                uint idx = u / BITS_PER_WORD;
                uint bit = (uint)1 << (int)(u % BITS_PER_WORD);
                bits[idx] |= bit;
            }
        }

        unsafe public void reset(uint u)
        {
            unchecked
            {
                uint idx = u / BITS_PER_WORD;
                uint bit = (uint)1 << (int)(u % BITS_PER_WORD);
                bits[idx] &= ~bit;
            }
        }

        unsafe public bool test(uint u)
        {
            unchecked
            {
                uint idx = u / BITS_PER_WORD;
                uint bit = u % BITS_PER_WORD;
                return ((bits[idx] >> (int)bit) & 1) > 0;
            }
        }

        public uint block(uint n)
        {
            uint idx = n / BITS_PER_WORD;
            return bits[idx];
        }
    };
}
