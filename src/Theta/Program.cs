// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Theta optimized miner is covered by the FAIR MINING license 2.1.1

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.IO.MemoryMappedFiles;

namespace Theta
{
    class Program
    {
        public static Queue<Solution> solutions = new Queue<Solution>();
        public static List<Tuple<uint, uint>> edges = new List<Tuple<uint, uint>>(150000);
        public static CGraph g = new CGraph();
        public static CGraph g2 = new CGraph();
        public static Task cycler;
        public static Process cuda;
        public volatile static TrimmerState tstate = TrimmerState.Starting;

        public static UInt64 k0, k1, k2, k3, nonce, reps;
        private static bool timeout;

        static void Main(string[] args)
        {
            try
            {
                if (!Directory.Exists("edges"))
                    Directory.CreateDirectory("edges");
            }
            catch { }

            System.Console.InputEncoding = System.Text.Encoding.ASCII;
            DateTime s = DateTime.Now;
            var parser = new SimpleCommandLineParser();
            parser.Parse(args);

            if (parser.Contains("n") && parser.Contains("r"))
            {
                try
                {
                    nonce = UInt64.Parse(parser.Arguments["n"][0]);
                    reps = UInt64.Parse(parser.Arguments["r"][0]);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Unable to parse arguments: " + ex.Message);
                }

                Console.WriteLine("Starting CUDA solver process...");

                try
                {
                    cuda = Process.Start(new ProcessStartInfo()
                    {
                        FileName = "Cudacka.exe",
                        CreateNoWindow = true,
                        RedirectStandardError = true,
                        RedirectStandardInput = true,
                        RedirectStandardOutput = true,
                        StandardErrorEncoding = Encoding.ASCII,
                        StandardOutputEncoding = Encoding.ASCII,
                        UseShellExecute = false
                    });

                    cuda.ErrorDataReceived += Cuda_ErrorDataReceived;
                    cuda.OutputDataReceived += Cuda_OutputDataReceived;
                    cuda.BeginOutputReadLine();
                    cuda.BeginErrorReadLine();

                    Task.Delay(2000).Wait();

                    if (!cuda.HasExited && tstate == TrimmerState.Ready)
                    {
                        s = DateTime.Now;

                        for (ulong i = 0; i < reps; i++)
                        {
                            Console.WriteLine("Trimming iteration " + i.ToString() + " of " + reps.ToString());
                            DateTime a = DateTime.Now;

                            GetSols();

                            byte[] header = new byte[80];
                            Array.Clear(header, 0, header.Length);
                            UInt32 hnonce = (uint)nonce + (uint)i;
                            var bytes = BitConverter.GetBytes(hnonce);
                            Array.Copy(bytes, 0, header, header.Length - 4, 4);
                            var hash = new Crypto.Blake2B(256);
                            byte[] blaked = hash.ComputeHash(header);

                            //UInt64 k0i = BitConverter.ToUInt64(blaked, 0);
                            //UInt64 k1i = BitConverter.ToUInt64(blaked, 8);

                            //k0 = k0i ^ 0x736f6d6570736575UL;
                            //k1 = k1i ^ 0x646f72616e646f6dUL;
                            //k2 = k0i ^ 0x6c7967656e657261UL;
                            //k3 = k1i ^ 0x7465646279746573UL;

                            k0 = BitConverter.ToUInt64(blaked, 0);
                            k1 = BitConverter.ToUInt64(blaked, 8);
                            k2 = BitConverter.ToUInt64(blaked, 16);
                            k3 = BitConverter.ToUInt64(blaked, 24);

                            cuda.StandardInput.WriteLine(string.Format("#t {0} {1} {2} {3} {4}", k0, k1, k2, k3, 0));

                            bool notify = false;
                            for (int w = 0; w < 500; w++)
                            {
                                if (tstate != TrimmerState.Trimming)
                                    Task.Delay(1).Wait();
                                else
                                    break;

                                if (w > 100 && !notify)
                                {
                                    notify = true;
                                    Console.WriteLine("Warning: No response from trimmer to trim command for > 100ms");
                                }
                            }

                            if (tstate == TrimmerState.Trimming)
                            {
                                try
                                {
                                    timeout = false;
                                    bool reported = false;

                                    for (int t = 0; t <= 10000; t++)
                                    {
                                        if (t == 10000)
                                        {
                                            timeout = true;
                                        }

                                        if (tstate != TrimmerState.Terminated)
                                        {
                                            if (tstate == TrimmerState.Ready && !reported)
                                            {
                                                reported = true;
                                                Console.ForegroundColor = ConsoleColor.Magenta;
                                                Console.WriteLine("Trimmed in " + Math.Round((DateTime.Now - a).TotalMilliseconds).ToString() + "ms");
                                                Console.ForegroundColor = ConsoleColor.White;
                                            }

                                            if (tstate == TrimmerState.Ready)
                                            {
                                                UInt64 _k0 = k0, _k1 = k1, _k2 = k2, _k3 = k3, _nonce = hnonce;
                                                if (cycler == null || cycler.IsCompleted)
                                                {
                                                    cycler = Task.Run(() =>
                                                    {
                                                        g.SetHeader(_nonce, _k0, _k1, _k2, _k3);
                                                        g.SetEdges(edges);
                                                        g.FindSolutions(42, solutions);
                                                    });
                                                }
                                                else
                                                {
                                                    cycler = Task.Run(() =>
                                                    {
                                                        g2.SetHeader(_nonce, _k0, _k1, _k2, _k3);
                                                        g2.SetEdges(edges);
                                                        g2.FindSolutions(42, solutions);
                                                    });
                                                }

                                                break;
                                            }

                                        }

                                        Task.Delay(1).Wait();
                                    }

                                    if (timeout)
                                    {
                                        Console.WriteLine("CUDA trimmer timeout");
                                        break;
                                    }

                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine("Error while waiting for trimmed edges: " + ex.Message);
                                }
                            }
                            else
                            {
                                Console.WriteLine("CUDA trimmer not responding");

                                try
                                {
                                    if (cuda.HasExited)
                                        Console.WriteLine("CUDA trimmer thread terminated itself with code " + cuda.ExitCode.ToString() );
                                    else
                                    {
                                        Console.WriteLine("CUDA trimmer stuck in " + tstate.ToString());
                                    }
                                }
                                catch { }

                                break;
                            }


                        }

                        if (cycler != null && !cycler.IsCompleted)
                            cycler.Wait();
                        if (g != null && g.recovery != null && !g.recovery.IsCompleted)
                            g.recovery.Wait();


                        if (solutions.Count > 0)
                        {
                            for (int i = 0; i < 10; i++)
                                if (tstate != TrimmerState.Ready)
                                    Task.Delay(200).Wait();

                            GetSols();

                            Task.Delay(500).Wait();
                        }


                        cuda.StandardInput.WriteLine("#e");
                    }
                    else
                        Console.WriteLine("CUDA launch error");
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error while starting CUDA solver: " + ex.Message);
                }
                finally
                {
                    try
                    {
                        if (cuda != null && !cuda.HasExited)
                            cuda.Kill();

                    }
                    catch { }

                }

            }
            else
            {
                Console.WriteLine("Please specify all options: -n <nonce> -r <iterations>");
            }





            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine("Finished in " + Math.Round((DateTime.Now - s).TotalMilliseconds).ToString() + "ms");
            Console.ForegroundColor = ConsoleColor.White;
        }

        public static void GetSols()
        {
            while (solutions.Count > 0)
            {
                var so = solutions.Dequeue();
                if (so.nonces.Count == 42)
                {
                    tstate = TrimmerState.Solving;

                    cuda.StandardInput.Write(string.Format("#s {0} {1} {2} {3} {4}", so.k0, so.k1, so.k2, so.k3, 0));
                    foreach (var n in so.nonces)
                        cuda.StandardInput.Write(" " + ((UInt64)n.Item1 | ((UInt64)n.Item2 << 32)).ToString());
                    cuda.StandardInput.WriteLine();

                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine(string.Format("Solution: nonce:{0:X} k0:{1:X} k1:{2:X} k2:{3:X} k3:{4:X}", so.nonce, so.k0, so.k1, so.k2, so.k3));
                    Console.ForegroundColor = ConsoleColor.White;

                    int max = 2000;
                    Task.Delay(50);
                    while (tstate != TrimmerState.Ready)
                    {
                        if (--max < 0) break;
                        Task.Delay(1).Wait();
                    }
                }
            }
        }

        private static void Cuda_ErrorDataReceived(object sender, DataReceivedEventArgs e)
        {
            try
            {
                if (e.Data != null && e.Data != "" && e.Data[0] == '#')
                {
                    // valid command line
                    switch (e.Data[1])
                    {
                        case 'a':
                            tstate = TrimmerState.Trimming;
                            break;
                        case 'r':
                            tstate = TrimmerState.Ready;
                            break;
                        case 'e':
                            tstate = TrimmerState.SendingEdges;
                            if (cycler != null && !cycler.IsCompleted)
                            {
                                Console.WriteLine("Warning, CPU bottleneck detected");
                            }

                            edges.Clear();

                            try
                            {
                                using (var mmf = MemoryMappedFile.OpenExisting("CuckoDataSend"))
                                {
                                    using (var mmfs = mmf.CreateViewStream(0, 8000000, MemoryMappedFileAccess.Read))
                                    {
                                        using (var br = new BinaryReader(mmfs))
                                        {
                                            var count = br.ReadUInt32();

                                            for (int i = 0; i < count; i++)
                                            {
                                                var a = br.ReadUInt32();
                                                var b = br.ReadUInt32();
                                                edges.Add(new Tuple<uint, uint>(a, b));
                                            }
                                        }
                                    }
                                }
                            }
                            catch
                            {
                                // error in shared data stuff, lets try secondary solution
                                try
                                {
                                    //edges\\data.bin
                                    if (File.Exists("edges/data.bin"))
                                    {
                                        byte[] data =  File.ReadAllBytes("edges/data.bin");
                                        File.Delete("edges/data.bin");
                                        using (MemoryStream ms = new MemoryStream(data))
                                        using (BinaryReader br = new BinaryReader(ms))
                                        {
                                            var count = br.ReadUInt32();

                                            for (int i = 0; i < count; i++)
                                            {
                                                var a = br.ReadUInt32();
                                                var b = br.ReadUInt32();
                                                edges.Add(new Tuple<uint, uint>(a, b));
                                            }
                                        }
                                    }
                                }
                                catch
                                {
                                    Console.WriteLine("Unable to get edges from trimmer!");
                                }
                            }

                            
                            break;
                        case 'x':
                            tstate = TrimmerState.Terminated;
                            break;
                        case 's':
                            try
                            {
                                var nonces = e.Data.Split(' ');
                                var sols = nonces.Skip(1).Select(n => uint.Parse(n)).OrderBy(n => n).ToList();

                                Console.ForegroundColor = ConsoleColor.Red;

                                if (sols != null && sols.Count > 0)
                                {
                                    Console.WriteLine(sols.Select(s => s.ToString("X")).Aggregate((current, next) => current + ", " + next));
                                }
                                Console.ForegroundColor = ConsoleColor.White;
                            }
                            catch
                            {
                                Console.WriteLine("Solution return failed.");
                            }
                            break;
                    }
                }
                else if (e.Data != null)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine(e.Data);
                    Console.ForegroundColor = ConsoleColor.White;
                }
            }
            catch
            {

            }
        }

        private static void Cuda_OutputDataReceived(object sender, DataReceivedEventArgs e)
        {

        }
    }

    public class Solution
    {
        public UInt64 k0, k1, k2, k3, nonce;
        public List<Tuple<uint, uint>> nonces = new List<Tuple<uint, uint>>();
    }

    public enum TrimmerState
    {
        Starting,
        Ready,
        Trimming,
        SendingEdges,
        Terminated,
        Solving
    }

    public class CGraph
    {
        public Dictionary<uint, uint> graphU = new Dictionary<uint, uint>(150000);
        public Dictionary<uint, uint> graphV = new Dictionary<uint, uint>(150000);

        private List<Tuple<uint, uint>> edges;
        private int maxlength = 8192;
        public Task recovery;

        private UInt64 nonce, k0, k1, k2, k3;

        public void SetEdges(List<Tuple<uint, uint>> edges)
        {
            graphU.Clear();
            graphV.Clear();
            this.edges = edges;
        }

        public void SetHeader(UInt64 snonce, UInt64 k0, UInt64 k1, UInt64 k2, UInt64 k3)
        {
            this.nonce = snonce;
            this.k0 = k0;
            this.k1 = k1;
            this.k2 = k2;
            this.k3 = k3;
        }

        internal void FindSolutions(int cyclen, Queue<Solution> solutions)
        {
            int dupes = 0;

            foreach (var e in edges)
            {
                {
                    if (graphU.ContainsKey(e.Item1) && graphU[e.Item1] == e.Item2)
                    {
                        Console.WriteLine("2-cycle found");
                        continue;
                    }

                    if (graphV.ContainsKey(e.Item2) && graphV[e.Item2] == e.Item1)
                    {
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
                            Console.WriteLine(cycle.ToString() + "-cycle found");
                        else if (cycle == cyclen)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine("42-cycle found!");
                            // initiate nonce recovery procedure
                            Console.ForegroundColor = ConsoleColor.White;

                            List<uint> path1t = path1.Take((int)joinA + 1).ToList();
                            List<uint> path2t = path2.Take((int)joinB + 1).ToList();
                            List<Tuple<uint, uint>> cycleEdges = new List<Tuple<uint, uint>>(42);
                            cycleEdges.Add(e);

                            // need list of the 42 edges as tuples here....

                            cycleEdges.AddRange(path1t.Zip(path1t.Skip(1), (second, first) => new Tuple<uint, uint>(first, second)));
                            cycleEdges.AddRange(path2t.Zip(path2t.Skip(1), (second, first) => new Tuple<uint, uint>(first, second)));

                            solutions.Enqueue(new Solution() { k0 = k0, k1 = k1, k2 = k2, k3 = k3, nonce = nonce, nonces = cycleEdges });
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

    //http://blog.gauffin.org/2014/12/simple-command-line-parser/
    public class SimpleCommandLineParser
    {
        public SimpleCommandLineParser()
        {
            Arguments = new Dictionary<string, string[]>();
        }
        public IDictionary<string, string[]> Arguments { get; private set; }
        public void Parse(string[] args)
        {
            var currentName = "";
            var values = new List<string>();
            foreach (var arg in args)
            {
                if (arg.StartsWith("-"))
                {
                    if (currentName != "")
                        Arguments[currentName] = values.ToArray();
                    values.Clear();
                    currentName = arg.Substring(1);
                }
                else if (currentName == "")
                    Arguments[arg] = new string[0];
                else
                    values.Add(arg);
            }
            if (currentName != "")
                Arguments[currentName] = values.ToArray();
        }
        public bool Contains(string name)
        {
            return Arguments.ContainsKey(name);
        }
    }

    public class Cucko30
    {
        public static Dictionary<UInt64, Tuple<uint, uint>> lookup = new Dictionary<UInt64, Tuple<uint, uint>>(150);
        public static Dictionary<UInt64, uint> nonces = new Dictionary<ulong, uint>(150);

        public static List<uint> RecoverSolution(List<Tuple<uint, uint>> edges, UInt64 startingNonce, UInt64 k0, UInt64 k1, UInt64 k2, UInt64 k3)
        {
            const uint max = 1 << 29;
            const uint edgemask = max - 1;
            uint threads = (uint)Math.Max(Environment.ProcessorCount / 2, 1);
            uint count = max / threads;

            lookup.Clear();
            nonces.Clear();
            for (int i = 0; i < edges.Count; i++)
            {
                var e = edges[i];
                lookup[(UInt64)e.Item1 | ((UInt64)e.Item2 << 32)] = e;
                lookup[(UInt64)e.Item2 | ((UInt64)e.Item1 << 32)] = e;
            }

            Parallel.For((uint)0, threads,
                   index => {
                       uint start = (uint)index * count;
                       for (uint i = start; i < (start + count); i++)
                       {
                           UInt64 u = (uint)siphash24(k0, k1, k2, k3, (startingNonce + i) * 2) & edgemask;
                           UInt64 v = (uint)siphash24(k0, k1, k2, k3, (startingNonce + i) * 2 + 1) & edgemask;

                           UInt64 a = u | (v << 32);
                           UInt64 b = v | (u << 32);

                           if (lookup.ContainsKey(a) || lookup.ContainsKey(b))
                               nonces[a] = i;

                       }
                   });

            var sols = nonces.Values.Distinct().OrderBy(n => n).ToList();
            Console.ForegroundColor = ConsoleColor.Red;

            if (sols != null && sols.Count > 0)
            {
                Console.Write("Solution: ");
                Console.WriteLine(sols.Select(s => s.ToString()).Aggregate((current, next) => current + ", " + next));
            }
            Console.ForegroundColor = ConsoleColor.White;
            return sols;
        }

        public static UInt64 ROTL(UInt64 x, byte b)
        {
            return (((x) << (b)) | ((x) >> (64 - (b))));
        }

        public static UInt64 siphash24(UInt64 k0, UInt64 k1, UInt64 k2, UInt64 k3, UInt64 nonce)
        {
            unchecked
            {
                UInt64 v0 = k0, v1 = k1, v2 = k2, v3 = k3 ^ nonce;

                //    SIPROUND; SIPROUND;
                v0 += v1; v2 += v3; v1 = ROTL(v1, 13);
                v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;
                v0 = ROTL(v0, 32); v2 += v1; v0 += v3;
                v1 = ROTL(v1, 17); v3 = ROTL(v3, 21);
                v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);

                v0 += v1; v2 += v3; v1 = ROTL(v1, 13);
                v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;
                v0 = ROTL(v0, 32); v2 += v1; v0 += v3;
                v1 = ROTL(v1, 17); v3 = ROTL(v3, 21);
                v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);


                v0 ^= nonce;
                v2 ^= 0xff;
                //SIPROUND; SIPROUND; SIPROUND; SIPROUND;

                v0 += v1; v2 += v3; v1 = ROTL(v1, 13);
                v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;
                v0 = ROTL(v0, 32); v2 += v1; v0 += v3;
                v1 = ROTL(v1, 17); v3 = ROTL(v3, 21);
                v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);

                v0 += v1; v2 += v3; v1 = ROTL(v1, 13);
                v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;
                v0 = ROTL(v0, 32); v2 += v1; v0 += v3;
                v1 = ROTL(v1, 17); v3 = ROTL(v3, 21);
                v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);

                v0 += v1; v2 += v3; v1 = ROTL(v1, 13);
                v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;
                v0 = ROTL(v0, 32); v2 += v1; v0 += v3;
                v1 = ROTL(v1, 17); v3 = ROTL(v3, 21);
                v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);

                v0 += v1; v2 += v3; v1 = ROTL(v1, 13);
                v3 = ROTL(v3, 16); v1 ^= v0; v3 ^= v2;
                v0 = ROTL(v0, 32); v2 += v1; v0 += v3;
                v1 = ROTL(v1, 17); v3 = ROTL(v3, 21);
                v1 ^= v2; v3 ^= v0; v2 = ROTL(v2, 32);


                return (v0 ^ v1) ^ (v2 ^ v3);
            }
        }
    }
}
