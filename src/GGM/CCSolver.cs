// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GGM
{

    public class Solution
    {
        public UInt64 k0, k1, k2, k3, nonce, height, difficulty, jobId;
        public List<Tuple<uint, uint>> nonces = new List<Tuple<uint, uint>>();
    }

    public class CGraph
    {
        const bool ShowCycles = false;

        public Dictionary<uint, uint> graphU;
        public Dictionary<uint, uint> graphV;

        private Tuple<uint, uint>[] edges;
        private int maxlength = 8192;
        public Task recovery;

        private UInt64 nonce, k0, k1, k2, k3, height, diff, jobId;

        public void SetEdges(List<Tuple<uint, uint>> edges)
        {
            lock (edges)
            {
                this.edges = edges.ToArray();
            }

            graphU = new Dictionary<uint, uint>(edges.Count);
            graphV = new Dictionary<uint, uint>(edges.Count);
        }

        public void SetHeader(UInt64 snonce, UInt64 k0, UInt64 k1, UInt64 k2, UInt64 k3, UInt64 height, UInt64 diff, UInt64 jobId)
        {
            this.nonce = snonce;
            this.k0 = k0;
            this.k1 = k1;
            this.k2 = k2;
            this.k3 = k3;
            this.height = height;
            this.diff = diff;
            this.jobId = jobId;
        }

        internal void FindSolutions(int cyclen, Queue<Solution> solutions)
        {
            int dupes = 0;

            foreach (var e in edges)
            {
                {
                    if (graphU.ContainsKey(e.Item1) && graphU[e.Item1] == e.Item2)
                    {
                        if (ShowCycles)
                            Console.WriteLine("2-cycle found");
                        continue;
                    }

                    if (graphV.ContainsKey(e.Item2) && graphV[e.Item2] == e.Item1)
                    {
                        if (ShowCycles)
                            Console.WriteLine("2-cycle found");
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
                                Console.WriteLine(cycle.ToString() + "-cycle found");
                        }
                        else if (cycle == cyclen)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine("42-cycle found!");
                            // initiate nonce recovery procedure
                            Console.ResetColor();

                            List<uint> path1t = path1.Take((int)joinA + 1).ToList();
                            List<uint> path2t = path2.Take((int)joinB + 1).ToList();
                            List<Tuple<uint, uint>> cycleEdges = new List<Tuple<uint, uint>>(42);
                            cycleEdges.Add(e);

                            // need list of the 42 edges as tuples here....

                            cycleEdges.AddRange(path1t.Zip(path1t.Skip(1), (second, first) => new Tuple<uint, uint>(first, second)));
                            cycleEdges.AddRange(path2t.Zip(path2t.Skip(1), (second, first) => new Tuple<uint, uint>(first, second)));

                            solutions.Enqueue(new Solution() { k0 = k0, k1 = k1, k2 = k2, k3 = k3, nonce = nonce, nonces = cycleEdges, height = height, difficulty = diff, jobId = jobId });
                            //recovery = Task.Run(() => { Cucko30.RecoverSolution(cycleEdges, snonce, k0,k1,k2,k3); });
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

            while (graph.ContainsKey(key))
            {
                uint v = graph[key];

                if (path.Count >= maxlength)
                    break;

                path.Add(v);

                startInGraphU = !startInGraphU;
                graph = startInGraphU ? graphU : graphV;

                key = v;
            }

            return path;
        }


    }
}
