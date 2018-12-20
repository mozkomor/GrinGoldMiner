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
        public volatile static int activeCyclers = 0;
        public static Process cuda;
        private static Solution ActiveSolution;
        public volatile static TrimmerState tstate = TrimmerState.Starting;
        public volatile static bool SupressCudaConsole = false;
        public static string CppInitText = "";

        public static UInt64 k0, k1, k2, k3, nonce, reps, device = 0;
        public static string node = "127.0.0.1";
        private static bool timeout;
        private static bool Canceled = false;
        public static bool Test = false;

        private static int trimfailed = 0;
        static Random rnd = new Random((int)DateTime.Now.Ticks);
        static Stats statistics = new Stats();

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

            if (parser.Contains("t"))
            {
                Test = true;
                /*
                 *  src/cuckaroo$ ./cuda29 -n 77
                    GeForce GTX 1070 with 8119MB @ 256 bits x 4004MHz
                    Looking for 42-cycle on cuckaroo29("",77) with 50% edges, 64*64 buckets, 176 trims, and 64 thread blocks.
                    Using 6976MB of global memory.
                    nonce 77 k0 k1 k2 k3 f4956dc403730b01 e6d45de39c2a5a3e cbf626a8afee35f6 4307b94b1a0c9980
                    Seeding completed in 355 + 131 ms
                    67962 edges after trimming
                      8-cycle found
                      12-cycle found
                      42-cycle found
                    findcycles edges 67962 time 121 ms total 1044 ms
                    Time: 1044 ms
                    Solution 7d86f6 30eca94 4c4e3b8 5fdc721 70dd206 737c0cd 7b3b464 7dfd358 9038cc2 913872c b0a40a6 b50ea02 b52718b b58c806 d3a1049 d4f4485 e1083cf e267035 e531581 eb1e9bf ef7c556 11037141 11b20da7 11c73af0 136af3d4 13a9f961 13b4b0d9 146a4fed 161015fe 16125cb0 1653304f 18157684 18c2fc5b 18d4a39e 1962c64d 1bb64237 1c46b245 1d40bc3d 1d49d47c 1d8a0e1d 1e80fdc8 1fb4541e
                    Verified with cyclehash 6d6545ca75f63e13c428a5e495e548e3c573b15af8841951599ba037d2f2ef58
                    1 total solutions
                 */
            }

            try
            {
                if (parser.Contains("a"))
                {
                    string remote = parser.Arguments["a"][0];
                    if (remote.Contains(":"))
                        gc = new GrinConeeect(remote.Split(':')[0], int.Parse(remote.Split(':')[1])); // user specified port
                    else
                        gc = new GrinConeeect(remote, 13416); // default port

                    gc.statistics = statistics;
                }
                else
                {
                    Console.WriteLine("Please specify options: [-d <cuda device>] -a <node IP>");
                    return;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error parsing remote address:port" + ex.Message);
            }


            if (!gc.IsConnected)
            {
                Console.WriteLine("Unable to connect to node.");
                return;
            }

            if (parser.Contains("l"))
            {
                string login = parser.Arguments["l"][0];
                string pwd = parser.Contains("p") ? parser.Arguments["p"][0] : "";

                gc.SendLogin(login, pwd);
            }

            Console.Write("Waiting for next block, this may take a bit");
            int wcnt = 0;
            while (true)
            {
                if (gc.CurrentJob != null && gc.IsConnected)
                    break;

                if (Canceled)
                    return;

                Console.Write(".");

                if ((++wcnt % 30) == 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("\nMake sure you are running 'grin wallet listen' on your node! \n");
                    Console.ResetColor();
                }

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
                    device = parser.Contains("d") ? UInt64.Parse(parser.Arguments["d"][0]) : 0;
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Unable to parse arguments: " + ex.Message);
                }

                Console.WriteLine("Starting CUDA solver process on device " + device.ToString());

                try
                {
                    bool lowMem = parser.Contains("lm");

                    if(lowMem)
                        Console.WriteLine("8GB version switch is deprecated");

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

                    Task.Delay(5000).Wait();

                    if (cuda.HasExited)
                        Console.WriteLine("C++ trimmer self-terminated!");

                    if (tstate != TrimmerState.Ready)
                    {
                        Console.WriteLine("C++ trimmer not in ready state!");

                        if (CppInitText.Trim() == "")
                        {
                            Console.WriteLine("Console output redirection failed! Lets try to use sockets for OpenCL miner instead...");
                        }
                        else
                        {
                            Console.WriteLine("Unexpected data from C++ (expected #r): " + CppInitText);
                        }
                    }

                    SupressCudaConsole = true;

                    if (!cuda.HasExited && tstate == TrimmerState.Ready)
                    {
                        s = DateTime.Now;

                        UInt64 jobId = 0;
                        UInt64 height = 0;
                        UInt64 dif = 0;
                        byte[] header;

                        //for (ulong i = 0; i < reps; i++)
                        while (!Canceled)
                        {
                            if (gc.lastComm.AddMinutes(30) < DateTime.Now)
                                gc.WaitForJob = true;

                            if (gc.WaitForJob | !gc.IsConnected)
                            {
                                Task.Delay(100).Wait();
                                Console.Write(".");
                                continue;
                            }

                            DateTime a = DateTime.Now;

                            GetSols();

                            jobId = gc.CurrentJob.job_id;
                            height = (UInt64)gc.CurrentJob.height;
                            dif = (UInt64)gc.CurrentJob.difficulty;
                            header = gc.CurrentJob.GetHeader();

                            UInt64 hnonce = (UInt64)(long)rnd.Next() | ((UInt64)(long)rnd.Next() << 32);
                            var bytes = BitConverter.GetBytes(hnonce).Reverse().ToArray();
                            //Array.Copy(bytes, 0, header, header.Length - 8, 8);
                            header = header.Concat(bytes).ToArray();
                            var hash = new Crypto.Blake2B(256);
                            byte[] blaked = hash.ComputeHash(header);
                            //blaked = hash.ComputeHash(blaked); -- testnet2 bug

                            if (Test)
                            {
                                k0 = 0xf4956dc403730b01L;
                                k1 = 0xe6d45de39c2a5a3eL;
                                k2 = 0xcbf626a8afee35f6L;
                                k3 = 0x4307b94b1a0c9980L;
                            }
                            else
                            {
                                k0 = BitConverter.ToUInt64(blaked, 0);
                                k1 = BitConverter.ToUInt64(blaked, 8);
                                k2 = BitConverter.ToUInt64(blaked, 16);
                                k3 = BitConverter.ToUInt64(blaked, 24);
                            }

                            if (statistics.graphs % 100 == 0)
                            {
                                Console.WriteLine("Graphs: {0}, Trims: {1}, Shares: {2}, Mined: {3}", statistics.graphs, statistics.edgesets, statistics.solutions, statistics.mined);
                                if (statistics.solutions > 0)
                                    Console.WriteLine("Graphs per Solution: {0}", statistics.graphs / statistics.solutions);
                                if (statistics.graphs > 0)
                                    Console.WriteLine("GPS(Graphs/Second): {0:F2}", (float)statistics.graphs/(0.1 + (DateTime.Now - s).TotalSeconds));
                            }

                            statistics.graphs++;
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
                                                trimfailed = 0;
                                                reported = true;
                                                Console.ForegroundColor = ConsoleColor.Magenta;
                                                Console.WriteLine("Trimmed in " + Math.Round((DateTime.Now - a).TotalMilliseconds).ToString() + "ms");
                                                Console.ResetColor();
                                            }

                                            if (tstate == TrimmerState.Ready)
                                            {
                                                UInt64 _k0 = k0, _k1 = k1, _k2 = k2, _k3 = k3, _nonce = hnonce;

                                                if (activeCyclers < 4)
                                                {
                                                    Task.Run(() =>
                                                    {
                                                        try
                                                        {
                                                            activeCyclers++;

                                                            statistics.edgesets++;
                                                            CGraph g = new CGraph();
                                                            g.SetHeader(_nonce, _k0, _k1, _k2, _k3, height, dif, jobId);
                                                            g.SetEdges(edges);
                                                            g.FindSolutions(42, solutions);
                                                        }
                                                        catch
                                                        {

                                                        }
                                                        finally
                                                        {
                                                            activeCyclers--;
                                                        }
                                                    });
                                                }
                                                else
                                                {
                                                    Console.WriteLine("CPU overloaded, dropping tasks, CPU bottleneck!");
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

                                if (trimfailed++ > 3)
                                    break;
                            }


                        }

                        if (activeCyclers > 0)
                            Task.Delay(500).Wait();

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
                if (e.Data != null && !SupressCudaConsole)
                    CppInitText += e.Data;

                if (e.Data != null && e.Data != "" && e.Data.Trim()[0] == '#')
                {
                    // valid command line
                    switch (e.Data.Trim()[1])
                    {
                        case 'a':
                            tstate = TrimmerState.Trimming;
                            break;
                        case 'r':
                            tstate = TrimmerState.Ready;
                            break;
                        case 'e':
                            tstate = TrimmerState.SendingEdges;

                            lock (edges)
                            {
                                edges.Clear();

                                try
                                {
                                    using (var mmf = MemoryMappedFile.OpenExisting("CDS"+device.ToString()))
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

                                if (!Test)
                                {
                                    if ((ulong)gc.CurrentJob.job_id == ActiveSolution.jobId)
                                        Task.Run(() => { gc.SendSolution(ActiveSolution, sols); });

                                    //var diffOk = CheckAdditionalDifficulty(sols, ActiveSolution.difficulty, out ulong diff);
                                    //if (diffOk && (ulong)gc.CurrentJob.job_id == ActiveSolution.jobId)
                                    //{
                                    //    Console.ForegroundColor = ConsoleColor.Red;
                                    //    Console.WriteLine("Solution difficulty: " + diff.ToString() + " | " + ActiveSolution.difficulty);
                                    //    Console.ResetColor();

                                    //    Task.Run(() => { gc.SendSolution(ActiveSolution, sols); });
                                    //}
                                    //else if ((ulong)gc.CurrentJob.job_id == ActiveSolution.jobId)
                                    //{
                                    //    Console.ForegroundColor = ConsoleColor.Green;
                                    //    Console.WriteLine("Solution difficulty: " + diff.ToString() + " | " + ActiveSolution.difficulty);
                                    //    Console.ResetColor();
                                    //}
                                }

                                statistics.solutions++;

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
            catch (Exception ex)
            {
                if (!SupressCudaConsole)
                {
                    Console.WriteLine("Unknown problem parsing c++ output: " + e.Data + ", " + ex.Message);
                }
            }
        }

        private static bool CheckAdditionalDifficulty(List<uint> sols, ulong target, out ulong diff)
        {
            var solB = sols.Select(x => BitConverter.GetBytes(x).Reverse().ToArray()).SelectMany(x => x).ToArray<Byte>();

            var hash = new Crypto.Blake2B(256);
            UInt64 blaked = BitConverter.ToUInt64( hash.ComputeHash(solB).Reverse().ToArray(), 24 );

            UInt64 div = (UInt64.MaxValue / blaked);

            diff = (ulong)div;

            return  div >= target;
        }

        private static void Cuda_OutputDataReceived(object sender, DataReceivedEventArgs e)
        {
            if (!SupressCudaConsole)
            {
                Console.WriteLine("Unexpected c++ stdout data: " + e.Data);
            }
        }
    }

    public class Solution
    {
        public UInt64 k0, k1, k2, k3, nonce, height, difficulty, jobId;
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
        private bool ShowCycles = Program.Test;

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
            //int dupes = 0;

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
