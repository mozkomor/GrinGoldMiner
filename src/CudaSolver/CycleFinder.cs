using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SharedSerialization;

namespace CudaSolver
{
    public static class FinderBag
    {
        private static ConcurrentBag<CGraph> finders = new ConcurrentBag<CGraph>();
        public volatile static int findersInFlight = 0;
        public static int finderTime = 0;
        public static volatile int trimRounds = 80;

        private static CGraph GetFinder()
        {
            if (finders.TryTake(out CGraph finder))
                return finder;
            else
            {
                return new CGraph();
            }
        }

        private static void ReturnFinder(CGraph finder)
        {
            findersInFlight--;

            if (finders.Count < 10)
                finders.Add(finder);
        }

        public static void RunFinder(bool TEST, ref int trims, uint edgesCount, int[] edgesLeft, Job currentJob, ConcurrentQueue<Solution> graphSolutions, Stopwatch sw)
        {
            CGraph cg = GetFinder();
            cg.SetEdges(edgesLeft, (int)edgesCount);
            cg.SetHeader(currentJob);

            var cancellationTokenSource = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));
            var finder = Task.Factory.StartNew(() =>
            {
                try
                {
                    if (edgesCount < 120000)
                    {
                        if (cg.edges.Count(e => e == 0) > 100)
                        {
                            //Console.ForegroundColor = ConsoleColor.Red;
                            //Console.WriteLine("edges corrupted");
                            //Console.ResetColor();
                            return;
                        }

                        if (findersInFlight++ < 3)
                        {
                            Stopwatch cycleTime = new Stopwatch();
                            cycleTime.Start();
                            cg.FindSolutions(graphSolutions);
                            cycleTime.Stop();
                            AdjustTrims(cycleTime.ElapsedMilliseconds);
                            if (TEST)
                            {
                                Logger.Log(LogLevel.Info, string.Format("Finder completed in {0}ms on {1} edges with {2} solution(s) and {3} dupes", sw.ElapsedMilliseconds, edgesCount, graphSolutions.Count, cg.dupes));
                            }
                        }
                        else
                            Logger.Log(LogLevel.Warning, string.Format("CPU overloaded / {0}ms on {1} edges", sw.ElapsedMilliseconds, edgesCount));
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log(LogLevel.Warning, "Cycle finder crashed " + ex.Message);
                }
                finally
                {
                    ReturnFinder(cg);
                }
            }, cancellationTokenSource.Token);

            //Task.Delay(2000).ContinueWith((t) => {
            //    if (finder != null && finder.Status != TaskStatus.RanToCompletion)
            //    {
            //        File.AppendAllText("overload.txt", $"{DateTime.Now.ToString()}: thread killed");
                    
            //        finder.Dispose();
            //    }
            //});
        }

        private static void AdjustTrims(long elapsedMilliseconds)
        {
            int target = Comms.cycleFinderTargetOverride > 0 ? Comms.cycleFinderTargetOverride : 20 * Environment.ProcessorCount;
            if (elapsedMilliseconds > target)
                trimRounds += 10;
            else
                trimRounds -= 10;

            trimRounds = Math.Max(80, trimRounds);
            trimRounds = Math.Min(300, trimRounds);
        }
    }

    /// <summary>
    /// Copy of cycle finder from John Tromp
    /// https://github.com/tromp/cuckoo/blob/master/src/cuckaroo
    /// </summary>
    public class CGraph
    {
        const uint EDGE_MASK = (1 << 29) - 1;
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

        internal int[] edges;
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
            try
            {
                cg.reset();
                cg.SetProof(job.type);
                int proofSize = job.GetProofSize();

                for (int ee = 0; ee < edges.Length / 2; ee++)
                {
                    cg.add_compress_edge((uint)edges[ee * 2 + 0] & EDGE_MASK, (uint)edges[ee * 2 + 1] & EDGE_MASK);
                }
                for (int s = 0; s < cg.nsols; s++)
                {
                    List<Edge> lst = new List<Edge>(proofSize);
                    // print_log("Solution");
                    for (int j = 0; j < proofSize; j++)
                    {
                         lst.Add(new Edge((uint)edges[cg.sols[s][j] * 2], (uint)edges[cg.sols[s][j] * 2 + 1]));
                    }

                    solutions.Enqueue(new Solution() { job = this.job, edges = lst });
                }
            }
            catch { // need to silent unknown rare cycle finder errors
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
        private const byte MAXPROOF = 42;
        public static uint PROOFSIZE = 42;

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
            links = new link[MAXEDGES];
            compressu = compressv = null;
            sharedmem = false;
            sols =  new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols[i] = new uint[MAXPROOF];
            visited = new bitmap(2 * maxnodes);
            visited.clear();
        }

        public graph(uint maxedges, uint maxnodes, uint maxsols, uint compressbits)// : visited(2*maxnodes)
        {
            MAXEDGES = maxedges;
            MAXNODES = maxnodes;
            MAXSOLS = maxsols;
            adjlist = new uint[2 * MAXNODES]; // index into links array
            links = new link[MAXEDGES];
            compressu = new compressor(EDGEBITS, compressbits);
            compressv = new compressor(EDGEBITS, compressbits);
            sharedmem = false;
            sols = new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols.Add(new uint[MAXPROOF]);
            visited = new bitmap(2 * maxnodes);
            visited.clear();
        }

        public graph(uint maxedges, uint maxnodes, uint maxsols, char* bytes)// : visited(2*maxnodes)
        {
            MAXEDGES = maxedges;
            MAXNODES = maxnodes;
            MAXSOLS = maxsols;
            adjlist = new uint[2 * MAXNODES]; // index into links array
            links = new link[MAXEDGES];
            compressu = compressv = null;
            sharedmem = true;
            sols = new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols.Add(new uint[MAXPROOF]);
            visited = new bitmap(2 * maxnodes);
            visited.clear();
        }

        public graph(uint maxedges, uint maxnodes, uint maxsols, uint compressbits, char* bytes)// : visited(2*maxnodes)
        {
            MAXEDGES = maxedges;
            MAXNODES = maxnodes;
            MAXSOLS = maxsols;
            adjlist = new uint[2 * MAXNODES]; // index into links array
            links = new link[MAXEDGES];
            compressu = new compressor(EDGEBITS, compressbits);
            compressv = new compressor(EDGEBITS, compressbits);
            sharedmem = true;
            sols = new List<uint[]>((int)maxsols);
            for (int i = 0; i < maxsols; i++) sols.Add(new uint[MAXPROOF]);
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
            return a==b;
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
                        if (PROOFSIZE != 42)
                            sols[(int)nsols] = sols[(int)nsols++].Take((int)PROOFSIZE).OrderBy(e => e).Concat(new uint[42-PROOFSIZE]).ToArray();
                        else
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
                        sols[(int)nsols][len] = au1;
                        cycles_with_link(len + 1, links[au1].to, dest);
                    }
                    visited.reset(u);
                }
            }
        }

        public void add_edge(uint u, uint v, uint dir)
        {
            unchecked
            {
                //assert(u < MAXNODES);
                //assert(v < MAXNODES);
                v += MAXNODES; // distinguish partitions
                if (dir != 0)
                {
                    uint tmp = v;
                    v = u;
                    u = tmp;
                }
                if (adjlist[v] != NIL)
                { // possibly part of a cycle
                    sols[(int)nsols][0] = nlinks;
                    cycles_with_link(1, v, u);
                }

                uint ulink = nlinks++;
                links[ulink].next = adjlist[u];
                links[adjlist[u] = ulink].to = v;
            }
        }

        public void add_compress_edge(uint u, uint v)
        {
            //add_edge(compressu.compress(u), compressv.compress(v));
            add_edge(compressu.compress(u) >> 1, compressv.compress(v) >> 1, u & 1);
        }

        internal void SetProof(CuckooType type)
        {
            graph.PROOFSIZE = (type == CuckooType.GRIN29) ? 42u : 32u;
        }
    };

    /// <summary>
    /// https://github.com/tromp/cuckoo/blob/master/src/cuckaroo
    /// FAIR MINING License
    /// </summary>
    unsafe public class compressor
    {
        public uint NODEBITS;
        public uint COMPRESSBITS;
        public uint SIZEBITS;
        public uint SIZEBITS1;
        public uint SIZE;
        public uint MASK;
        public uint MASK1;
        public uint npairs;
        public const uint NIL = ~(uint)0;
        public uint[] nodes;
        public bool sharedmem;

        public compressor(uint nodebits, uint compressbits, char* bytes)
        {
            NODEBITS = nodebits;
            COMPRESSBITS = compressbits;
            SIZEBITS = NODEBITS - COMPRESSBITS;
            SIZEBITS1 = SIZEBITS - 1;
            SIZE = (uint)1 << (int)SIZEBITS;
            MASK = SIZE - 1;
            MASK1 = MASK >> 1;
            nodes = new uint[SIZE];
            sharedmem = true;
            MASK = SIZE - 1;
        }

        public compressor(uint nodebits, uint compressbits)
        {
            NODEBITS = nodebits;
            COMPRESSBITS = compressbits;
            SIZEBITS = NODEBITS - COMPRESSBITS;
            SIZEBITS1 = SIZEBITS - 1;
            SIZE = (uint)1 << (int)SIZEBITS;
            MASK = SIZE - 1;
            MASK1 = MASK >> 1;
            nodes = new uint[SIZE];
            sharedmem = false;
            MASK = SIZE - 1;
        }

        //~compressor()
        //{
        //    if (!sharedmem)
        //        nodes = null;
        //}

        public UInt64 bytes()
        {
            return sizeof(uint) * SIZE;
        }

        public void reset()
        {
            //memset(nodes, (char)NIL, sizeof(uint[SIZE2]));
            Array.Fill(nodes, NIL);
            npairs = 0;
        }

        public uint compress(uint u)
        {
            unchecked
            {
                uint parity = u & 1;
                uint ui = u >> (int)COMPRESSBITS;
                u >>= 1;
                for (; ; ui = (ui + 1) & MASK)
                {
                    uint cu = nodes[ui];
                    if (cu == NIL)
                    {
                        if (npairs >= SIZE/2)
                        {
                            //print_log("NODE OVERFLOW at %x\n", u);
                            return parity;
                        }
                        nodes[ui] = u << (int)SIZEBITS1 | npairs;
                        return (npairs++ << 1) | parity;
                    }
                    if ((cu & ~MASK1) == u << (int)SIZEBITS1)
                    {
                        return ((cu & MASK1) << 1) | parity;
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
        //~bitmap()
        //{
        //    freebits();
        //}
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
