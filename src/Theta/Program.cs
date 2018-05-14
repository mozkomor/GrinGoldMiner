// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Theta optimized miner is covered by the FAIR MINING license

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.IO.MemoryMappedFiles;
using System.Numerics;

namespace Theta
{
    class Program
    {
        static GrinConeeect gc;

        public static Queue<Solution> solutions = new Queue<Solution>();
        public static List<Tuple<uint, uint>> edges = new List<Tuple<uint, uint>>(150000);
        //public static CGraph g = new CGraph();
        //public static CGraph g2 = new CGraph();
        public static Task cycler;
        public static Process cuda;
        private static Solution ActiveSolution;
        public volatile static TrimmerState tstate = TrimmerState.Starting;

        public static UInt64 k0, k1, k2, k3, nonce, reps, device = 0;
        public static string node = "127.0.0.1";
        private static bool timeout;
        private static bool Canceled = false;

        static Random rnd = new Random((int)DateTime.Now.Ticks);

        static void Main(string[] args)
        {
            Console.CancelKeyPress += delegate {
                Console.WriteLine("Ctrl+C - Exitting");
                if (gc != null && gc.IsConnected)
                    gc.GrinClose();
                if (cuda != null && !cuda.HasExited)
                    cuda.Kill();

                Environment.Exit(0);
            };

            System.Console.InputEncoding = System.Text.Encoding.ASCII;
            DateTime s = DateTime.Now;
            var parser = new SimpleCommandLineParser();
            parser.Parse(args);

            if (parser.Contains("a"))
            {
                gc = new GrinConeeect(parser.Arguments["a"][0], 13416);
            }
            else
            {
                Console.WriteLine("Please specify options: [-d <cuda device>] -a <node IP>");
                return;
            }

            if (gc.IsConnected)
                Console.WriteLine("Connected to node.");
            else
            {
                Console.WriteLine("Unable to connect to node.");
                return;
            }

            Console.Write("Waiting for job");
            while (true)
            {
                if (gc.CurrentJob != null && gc.IsConnected)
                    break;

                if (Canceled)
                    return;

                Console.Write(".");
                Task.Delay(500).Wait();
            }
            Console.WriteLine();

            try
            {
                if (!Directory.Exists("edges"))
                    Directory.CreateDirectory("edges");
            }
            catch { }


            {
                try
                {
                    nonce = 0;
                    reps = 0;
                    device = UInt64.Parse(parser.Arguments["d"][0]);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Unable to parse arguments: " + ex.Message);
                }

                Console.WriteLine("Starting CUDA solver process on device " + device.ToString());

                try
                {
                    cuda = Process.Start(new ProcessStartInfo()
                    {
                        FileName = "Cudacka.exe",
                        Arguments = device.ToString(),
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

                        UInt64 height = 0;
                        UInt64 dif = 0;
                        byte[] header;

                        //for (ulong i = 0; i < reps; i++)
                        while (!Canceled)
                        {

                            DateTime a = DateTime.Now;

                            GetSols();

                            height = (UInt64)gc.CurrentJob.height;
                            dif = (UInt64)gc.CurrentJob.difficulty;
                            header = gc.CurrentJob.GetHeader();

                            //string hh = "000100000000000100d3cab77958344e8b143f79f201d805fe2a12aa486c5304c66c9996e68b5bab8436000000005af94f2b0000000000015bf53db905b0d5ad8eb2e521420ea237c1a6096c53987db3d707e66945cda820b8efb36d954aa49eb76d167242e8415058523c4dbd3991d88b17b682c757a15893ca6d776586d14985884ef96d32880caec84259a8ca970718d864db981abe1d3eac4fd21ccaac43ded1c67a51d106b2b52b266784fc9919d46e11bc24afcb702e93";
                            //header = Enumerable.Range(0, hh.Length)
                            // .Where(x => x % 2 == 0)
                            // .Select(x => Convert.ToByte(hh.Substring(x, 2), 16))
                            // .ToArray();

                            UInt64 hnonce = (UInt64)rnd.Next() | ((UInt64)rnd.Next() << 32);
                            var bytes = BitConverter.GetBytes(hnonce).Reverse().ToArray();
                            //Array.Copy(bytes, 0, header, header.Length - 8, 8);
                            header = header.Concat(bytes).ToArray();
                            var hash = new Crypto.Blake2B(256);
                            byte[] blaked = hash.ComputeHash(header);
                            blaked = hash.ComputeHash(blaked);

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
                                                Console.ResetColor();
                                            }

                                            if (tstate == TrimmerState.Ready)
                                            {
                                                UInt64 _k0 = k0, _k1 = k1, _k2 = k2, _k3 = k3, _nonce = hnonce;
                                                
                                                {
                                                    Task.Run(() =>
                                                    {
                                                        CGraph g = new CGraph();
                                                        g.SetHeader(_nonce, _k0, _k1, _k2, _k3, height, dif);
                                                        g.SetEdges(edges);
                                                        g.FindSolutions(42, solutions);
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
                        //if (g != null && g.recovery != null && !g.recovery.IsCompleted)
                        //    g.recovery.Wait();


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


            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine("Finished in " + Math.Round((DateTime.Now - s).TotalMilliseconds).ToString() + "ms");
            Console.ResetColor();
        }

        public static void GetSols()
        {
            while (solutions.Count > 0)
            {
                var so = solutions.Dequeue();
                if (so.nonces.Count == 42)
                {
                    ActiveSolution = so;

                    tstate = TrimmerState.Solving;

                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine(string.Format("Recovering solution: nonce:{0:X} k0:{1:X} k1:{2:X} k2:{3:X} k3:{4:X}", so.nonce, so.k0, so.k1, so.k2, so.k3));
                    Console.ResetColor();

                    cuda.StandardInput.Write(string.Format("#s {0} {1} {2} {3} {4}", so.k0, so.k1, so.k2, so.k3, 0));
                    foreach (var n in so.nonces)
                        cuda.StandardInput.Write(" " + ((UInt64)n.Item1 | ((UInt64)n.Item2 << 32)).ToString());
                    cuda.StandardInput.WriteLine();

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

                            lock (edges)
                            {
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

                                var diffOk = CheckAdditionalDifficulty(sols, ActiveSolution.difficulty, out ulong diff);
                                if (diffOk && (ulong)gc.CurrentJob.height == ActiveSolution.height)
                                {
                                    Console.ForegroundColor = ConsoleColor.Green;
                                    Console.WriteLine("Solution difficulty at network: " + diff.ToString() + " @ " + ActiveSolution.difficulty);
                                    Console.ResetColor();
                                }
                                else if ((ulong)gc.CurrentJob.height == ActiveSolution.height)
                                {
                                    Console.ForegroundColor = ConsoleColor.Green;
                                    Console.WriteLine("Solution difficulty below network: " + diff.ToString() + " under " + ActiveSolution.difficulty);
                                    Console.ResetColor();
                                }

                                Task.Run(() => { gc.SendSolution(ActiveSolution, sols); });

                                //Console.ForegroundColor = ConsoleColor.Red;

                                //if (sols != null && sols.Count > 0)
                                //{
                                //    Console.WriteLine(sols.Select(s => s.ToString("X")).Aggregate((current, next) => current + ", " + next));
                                //}
                                //Console.ResetColor();
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
                    Console.ResetColor();
                }
            }
            catch
            {

            }
        }

        private static bool CheckAdditionalDifficulty(List<uint> sols, ulong target, out ulong diff)
        {
            var solB = sols.Select(x => BitConverter.GetBytes(x)).SelectMany(x => x).ToArray<Byte>();

            var hash = new Crypto.Blake2B(256);
            byte[] blaked = hash.ComputeHash(solB);
            blaked = blaked.Append<byte>(0).ToArray();

            var blakedBI = new BigInteger(blaked);
            byte[] maxb = new byte[33];
            Array.Fill<byte>(maxb, 255);
            maxb[32] = 0;

            var max256 = new BigInteger(maxb);

            BigInteger div = (max256 / blakedBI);

            diff = (ulong)div;

            return  div >= target;
        }

        private static void Cuda_OutputDataReceived(object sender, DataReceivedEventArgs e)
        {

        }
    }

    public class Solution
    {
        public UInt64 k0, k1, k2, k3, nonce, height, difficulty;
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
        const bool ShowCycles = false;

        public Dictionary<uint, uint> graphU;
        public Dictionary<uint, uint> graphV;

        private Tuple<uint, uint>[] edges;
        private int maxlength = 8192;
        public Task recovery;

        private UInt64 nonce, k0, k1, k2, k3, height, diff;

        public void SetEdges(List<Tuple<uint, uint>> edges)
        {
            lock (edges)
            {
                this.edges = edges.ToArray();
            }

            graphU = new Dictionary<uint, uint>(edges.Count);
            graphV = new Dictionary<uint, uint>(edges.Count);
        }

        public void SetHeader(UInt64 snonce, UInt64 k0, UInt64 k1, UInt64 k2, UInt64 k3, UInt64 height, UInt64 diff)
        {
            this.nonce = snonce;
            this.k0 = k0;
            this.k1 = k1;
            this.k2 = k2;
            this.k3 = k3;
            this.height = height;
            this.diff = diff;
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

                            solutions.Enqueue(new Solution() { k0 = k0, k1 = k1, k2 = k2, k3 = k3, nonce = nonce, nonces = cycleEdges, height = height, difficulty = diff });
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


}
