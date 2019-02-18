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


    public class CGraph
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

        public CGraph()
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

}
