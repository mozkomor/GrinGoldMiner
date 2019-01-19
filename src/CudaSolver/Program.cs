using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using SharedSerialization;
using System.Collections.Concurrent;

// dotnet publish -c Release -r win-x64

namespace CudaSolver
{
    class Program
    {
        const long DUCK_SIZE_A = 129L;
        const long DUCK_SIZE_B = 82L;
        const long BUFFER_SIZE_A = DUCK_SIZE_A * 1024 * 4096;
        const long BUFFER_SIZE_B = DUCK_SIZE_B * 1024 * 4096 / 2;
        const long BUFFER_SIZE_U32 = (DUCK_SIZE_A + DUCK_SIZE_B / 2) * 1024 * 4096;
        const long DUCK_EDGES_A = DUCK_SIZE_A * 1024;
        const long DUCK_EDGES_B = DUCK_SIZE_B * 1024;

        const long INDEX_SIZE = 64 * 64 * 4;

        static DeviceFamily d_Family = DeviceFamily.Other;
        static int deviceID = 0;
        static int port = 13500;
        static bool TEST = false;
        static volatile int trims = 0;
        static volatile int solutions = 0;
        static volatile int findersInFlight = 0;

        static CudaContext ctx;
        static CudaKernel meanSeedA;
        static CudaKernel meanSeedB;
        static CudaKernel meanSeedB_4;
        static CudaKernel meanRound;
        static CudaKernel meanRound_2;
        static CudaKernel meanTail;
        static CudaKernel meanRecover;
        static CudaKernel meanRoundJoin;

        static CudaDeviceVariable<ulong> d_buffer;
        static CudaDeviceVariable<ulong> d_bufferMid;
        static CudaDeviceVariable<ulong> d_bufferB;
        static CudaDeviceVariable<UInt32> d_indexesA;
        static CudaDeviceVariable<UInt32> d_indexesB;

        static CudaStream streamPrimary;
        static CudaStream streamSecondary;
        static int[] h_a = null;
        static CudaPageLockedHostMemory<int> hAligned_a = null;

        static UInt32[] h_indexesA = new UInt32[INDEX_SIZE*2];
        static UInt32[] h_indexesB = new UInt32[INDEX_SIZE*2];

        static Job currentJob;
        static Job nextJob;
        static Stopwatch timer = new Stopwatch();

        public static ConcurrentQueue<Solution> graphSolutions = new ConcurrentQueue<Solution>();

        private static int gpuCount = 0;
        private static bool fastCuda = false;
        private static volatile int trimRounds = 80;
        const string TestPrePow = "0001000000000000202e000000005c2e43ce014ca55dc4e0dffe987ee3eef9ca78e517f5ae7383c40797a4e8a9dd75ddf57eafac5471135202aa6054a2cc66aa5510ebdd58edcda0662a9e02d8232a4c90e90b7bddec1f32031d2894d76e3c390fc12b2dcc7a6f12b52be1d7aea70eac7b8ae0dc3f0ffb267e39b95a77e44e66d523399312a812d538afd00c7fd87275f4be7ef2f447ca918435d537c3db3c1d3e5d4f3b830432e5a283fab48917a5695324a319860a329cb1f6d1520ad0078c0f1dd9147f347f4c34e26d3063f117858d75000000000000babd0000000000007f23000000001ac67b3b00000155";

        static void Main(string[] args)
        {
            try
            {
                if (args.Length > 0)
                    deviceID = int.Parse(args[0]);
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Device ID parse error");
            }

            try
            {
                if (args.Length > 1)
                {
                    port = int.Parse(args[1]);
                    Comms.ConnectToMaster(port);
                }
                else
                {
                    TEST = true;
                    Logger.CopyToConsole = true;
                    CGraph.ShowCycles = true;
                }
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Master connection error");
            }

            try
            {
                if (args.Length > 3)
                {
                    gpuCount = int.Parse(args[3]);
                    fastCuda = gpuCount <= (Environment.ProcessorCount / 2);
                    if (fastCuda)
                        Logger.Log(LogLevel.Info, "Using single GPU blocking mode");
                }
            }
            catch
            {
                
            } 

            if (TEST)
            {
                currentJob = nextJob = new Job()
                {
                    jobID = 0,
                    k0 = 0xf4956dc403730b01L,
                    k1 = 0xe6d45de39c2a5a3eL,
                    k2 = 0xcbf626a8afee35f6L,
                    k3 = 0x4307b94b1a0c9980L,
                    pre_pow = TestPrePow,
                    timestamp = DateTime.Now
                };
            }
            else
            {
                currentJob = nextJob = new Job()
                {
                    jobID = 0,
                    k0 = 0xf4956dc403730b01L,
                    k1 = 0xe6d45de39c2a5a3eL,
                    k2 = 0xcbf626a8afee35f6L,
                    k3 = 0x4307b94b1a0c9980L,
                    pre_pow = TestPrePow,
                    timestamp = DateTime.Now
                };

                if (!Comms.IsConnected())
                {
                    Console.WriteLine("Master connection failed, aborting");
                    Logger.Log(LogLevel.Error, "No master connection, exitting!");
                    return;
                }

                if (deviceID < 0)
                {
                    int devCnt = CudaContext.GetDeviceCount();
                    GpuDevicesMessage gpum = new GpuDevicesMessage() { devices = new List<GpuDevice>(devCnt) };
                    for (int i = 0; i < devCnt; i++)
                    {
                        string name = CudaContext.GetDeviceName(i);
                        var info = CudaContext.GetDeviceInfo(i);
                        gpum.devices.Add(new GpuDevice() {deviceID = i, name = name, memory = info.TotalGlobalMemory });
                    }
                    //Console.WriteLine(devCnt);
                    Comms.gpuMsg = gpum;
                    Comms.SetEvent();
                    //Console.WriteLine("event fired");
                    Task.Delay(1000).Wait();
                    //Console.WriteLine("closing");
                    Comms.Close();
                    return;
                }
            }


            try
            {
                var assembly = Assembly.GetEntryAssembly();
                var resourceStream = assembly.GetManifestResourceStream("CudaSolver.kernel_x64.ptx");
                ctx = new CudaContext(deviceID, !fastCuda ? (CUCtxFlags.BlockingSync | CUCtxFlags.MapHost) : CUCtxFlags.MapHost);

                meanSeedA = ctx.LoadKernelPTX(resourceStream, "FluffySeed2A");
                meanSeedA.BlockDimensions = 128;
                meanSeedA.GridDimensions = 2048;
                meanSeedA.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanSeedB = ctx.LoadKernelPTX(resourceStream, "FluffySeed2B");
                meanSeedB.BlockDimensions = 128;
                meanSeedB.GridDimensions = 1024;
                meanSeedB.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanSeedB_4 = ctx.LoadKernelPTX(resourceStream, "FluffySeed2B");
                meanSeedB_4.BlockDimensions = 128;
                meanSeedB_4.GridDimensions = 512;
                meanSeedB_4.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanRound = ctx.LoadKernelPTX(resourceStream, "FluffyRound");
                meanRound.BlockDimensions = 512;
                meanRound.GridDimensions = 4096;
                meanRound.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanRound_2 = ctx.LoadKernelPTX(resourceStream, "FluffyRound");
                meanRound_2.BlockDimensions = 512;
                meanRound_2.GridDimensions = 2048;
                meanRound_2.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanRoundJoin = ctx.LoadKernelPTX(resourceStream, "FluffyRound_J");
                meanRoundJoin.BlockDimensions = 512;
                meanRoundJoin.GridDimensions = 4096;
                meanRoundJoin.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanTail = ctx.LoadKernelPTX(resourceStream, "FluffyTail");
                meanTail.BlockDimensions = 1024;
                meanTail.GridDimensions = 4096;
                meanTail.PreferredSharedMemoryCarveout = CUshared_carveout.MaxL1;

                meanRecover = ctx.LoadKernelPTX(resourceStream, "FluffyRecovery");
                meanRecover.BlockDimensions = 256;
                meanRecover.GridDimensions = 2048;
                meanRecover.PreferredSharedMemoryCarveout = CUshared_carveout.MaxL1;

            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Unable to create kernels: " + ex.Message);
                Task.Delay(500).Wait();
                Comms.Close();
                return;
            }

            try
            {
                d_buffer = new CudaDeviceVariable<ulong>(BUFFER_SIZE_U32);
                d_bufferMid = new CudaDeviceVariable<ulong>(d_buffer.DevicePointer + (BUFFER_SIZE_B * 8));
                d_bufferB = new CudaDeviceVariable<ulong>(d_buffer.DevicePointer + (BUFFER_SIZE_A * 8));

                d_indexesA = new CudaDeviceVariable<uint>(INDEX_SIZE * 2);
                d_indexesB = new CudaDeviceVariable<uint>(INDEX_SIZE * 2);

                Array.Clear(h_indexesA, 0, h_indexesA.Length);
                Array.Clear(h_indexesB, 0, h_indexesA.Length);

                d_indexesA = h_indexesA;
                d_indexesB = h_indexesB;

                streamPrimary = new CudaStream(CUStreamFlags.NonBlocking);
                streamSecondary = new CudaStream(CUStreamFlags.NonBlocking);
            }
            catch (Exception ex)
            {
                Task.Delay(200).Wait();
                Logger.Log(LogLevel.Error, $"Out of video memory! Only {ctx.GetFreeDeviceMemorySize()} free" );
                Task.Delay(500).Wait();
                Comms.Close();
                return;
            }

            try
            {
                AllocateHostMemory(true, ref h_a, ref hAligned_a, 1024*1024*32);
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Unable to create pinned memory.");
                Task.Delay(500).Wait();
                Comms.Close();
                return;
            }

            int loopCnt = 0;

            while (!Comms.IsTerminated)
            {
                try
                {
                    if (!TEST && (Comms.nextJob.pre_pow == null || Comms.nextJob.pre_pow == "" || Comms.nextJob.pre_pow == TestPrePow))
                    {
                        Logger.Log(LogLevel.Info, string.Format("Waiting for job...."));
                        Task.Delay(1000).Wait();
                        continue;
                    }

                    if (!TEST && ((currentJob.pre_pow != Comms.nextJob.pre_pow) || (currentJob.origin != Comms.nextJob.origin)))
                    {
                        currentJob = Comms.nextJob;
                        currentJob.timestamp = DateTime.Now;
                    }

                    if (!TEST && (currentJob.timestamp.AddMinutes(30) < DateTime.Now ) && Comms.lastIncoming.AddMinutes(30) < DateTime.Now)
                    {
                        Logger.Log(LogLevel.Info, string.Format("Job too old..."));
                        Task.Delay(1000).Wait();
                        continue;
                    }

                    // test runs only once
                    if (TEST && loopCnt++ > 100)
                        Comms.IsTerminated = true;

                    Solution s;
                    while (graphSolutions.TryDequeue(out s))
                    {
                        meanRecover.SetConstantVariable<ulong>("recovery", s.GetUlongEdges());
                        d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                        meanRecover.RunAsync(streamPrimary.Stream, s.job.k0, s.job.k1, s.job.k2, s.job.k3, d_indexesB.DevicePointer);
                        streamPrimary.Synchronize();
                        s.nonces = new uint[42];
                        d_indexesB.CopyToHost(s.nonces, 0, 0, 42 * 4);
                        s.nonces = s.nonces.OrderBy(n => n).ToArray();
                        lock (Comms.graphSolutionsOut)
                        {
                            Comms.graphSolutionsOut.Enqueue(s);
                        }
                        Comms.SetEvent();
                    }

                    currentJob = currentJob.Next();

                    Logger.Log(LogLevel.Debug, string.Format("GPU NV{4}:Trimming #{4}: {0} {1} {2} {3}", currentJob.k0, currentJob.k1, currentJob.k2, currentJob.k3, currentJob.jobID, deviceID));

                    timer.Restart();

                    d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);

                    meanSeedA.RunAsync(streamPrimary.Stream, currentJob.k0, currentJob.k1, currentJob.k2, currentJob.k3, d_bufferMid.DevicePointer, d_indexesB.DevicePointer);
                    meanSeedB_4.RunAsync(streamPrimary.Stream, d_bufferMid.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, 0);
                    meanSeedB_4.RunAsync(streamPrimary.Stream, d_bufferMid.DevicePointer, d_buffer.DevicePointer + ((BUFFER_SIZE_A * 8) / 4) * 1, d_indexesB.DevicePointer, d_indexesA.DevicePointer, 16);
                    meanSeedB_4.RunAsync(streamPrimary.Stream, d_bufferMid.DevicePointer, d_buffer.DevicePointer + ((BUFFER_SIZE_A * 8) / 4) * 2, d_indexesB.DevicePointer, d_indexesA.DevicePointer, 32);
                    meanSeedB_4.RunAsync(streamPrimary.Stream, d_bufferMid.DevicePointer, d_buffer.DevicePointer + ((BUFFER_SIZE_A * 8) / 4) * 3, d_indexesB.DevicePointer, d_indexesA.DevicePointer, 48);

                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                    meanRound_2.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer + ((BUFFER_SIZE_A * 8) / 4) * 2, d_bufferB.DevicePointer, d_indexesA.DevicePointer + (2048 * 4), d_indexesB.DevicePointer + (4096 * 4), DUCK_EDGES_A, DUCK_EDGES_B / 2);
                    meanRound_2.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer, d_bufferB.DevicePointer - (BUFFER_SIZE_B * 8), d_indexesA.DevicePointer, d_indexesB.DevicePointer, DUCK_EDGES_A, DUCK_EDGES_B / 2);
                    d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    meanRoundJoin.RunAsync(streamPrimary.Stream, d_bufferB.DevicePointer - (BUFFER_SIZE_B * 8), d_bufferB.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, DUCK_EDGES_B / 2, DUCK_EDGES_B / 2);

                    //d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    //meanRound.RunAsync(streamPrimary.Stream, d_bufferB.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, DUCK_EDGES_B, DUCK_EDGES_B / 2);
                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer, d_bufferB.DevicePointer, d_indexesA.DevicePointer, d_indexesB.DevicePointer, DUCK_EDGES_B / 2, DUCK_EDGES_B / 2);
                    d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_bufferB.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, DUCK_EDGES_B / 2, DUCK_EDGES_B / 2);
                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer, d_bufferB.DevicePointer, d_indexesA.DevicePointer, d_indexesB.DevicePointer, DUCK_EDGES_B / 2, DUCK_EDGES_B / 4);

                    for (int i = 0; i < trimRounds; i++)
                    {
                        d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                        meanRound.RunAsync(streamPrimary.Stream, d_bufferB.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, DUCK_EDGES_B / 4, DUCK_EDGES_B / 4);
                        d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                        meanRound.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer, d_bufferB.DevicePointer, d_indexesA.DevicePointer, d_indexesB.DevicePointer, DUCK_EDGES_B / 4, DUCK_EDGES_B / 4);
                    }

                    d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    meanTail.RunAsync(streamPrimary.Stream, d_bufferB.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer);

                    ctx.Synchronize();
                    streamPrimary.Synchronize();

                    uint[] count = new uint[2];
                    d_indexesA.CopyToHost(count, 0, 0, 8);

                    if (count[0] > 4194304)
                    {
                        // trouble
                        count[0] = 4194304;
                        // log
                    }

                    hAligned_a.AsyncCopyFromDevice(d_buffer.DevicePointer, 0, 0, count[0] * 8, streamPrimary.Stream);
                    streamPrimary.Synchronize();
                    System.Runtime.InteropServices.Marshal.Copy(hAligned_a.PinnedHostPointer, h_a, 0, ((int)count[0] * 8) / sizeof(int));

                    timer.Stop();
                    currentJob.solvedAt = DateTime.Now;
                    currentJob.trimTime = timer.ElapsedMilliseconds;

                    //Console.WriteLine("Trimmed in {0}ms to {1} edges", timer.ElapsedMilliseconds, count[0]);
                    Logger.Log(LogLevel.Info, string.Format("GPU NV{2}:     Trimmed in {0}ms to {1} edges", timer.ElapsedMilliseconds, count[0], deviceID));

                    if (TEST)
                    {
                        //Console.WriteLine("Trimmed in {0}ms to {1} edges", timer.ElapsedMilliseconds, count[0]);

                        CGraph cg = new CGraph();
                        cg.SetEdges(h_a, (int)count[0]);
                        cg.SetHeader(currentJob);

                        //currentJob = currentJob.Next();

                        Task.Factory.StartNew(() =>
                           {
                               Stopwatch sw = new Stopwatch();
                               sw.Start();

                               if (count[0] < 200000)
                               {
                                   try
                                   {
                                       if (findersInFlight++ < 3)
                                       {
                                           Stopwatch cycleTime = new Stopwatch();
                                           cycleTime.Start();
                                           cg.FindSolutions(graphSolutions);
                                           cycleTime.Stop();
                                           AdjustTrims(cycleTime.ElapsedMilliseconds);
                                           if (graphSolutions.Count > 0) solutions++;
                                       }
                                       else
                                           Logger.Log(LogLevel.Warning, "CPU overloaded!");
                                   }
                                   catch (Exception ex)
                                   {
                                       Logger.Log(LogLevel.Error, "Cycle finder error" + ex.Message);
                                   }
                                   finally
                                   {
                                       findersInFlight--;
                                   }
                               }

                               sw.Stop();

                               if (++trims % 50 == 0)
                               {
                                   Console.ForegroundColor = ConsoleColor.Green;
                                   Console.WriteLine("SOLS: {0}/{1} - RATE: {2:F1}", solutions, trims, (float)trims/solutions );
                                   Console.ResetColor();
                               }
                               //Console.WriteLine("Finder completed in {0}ms on {1} edges with {2} solution(s)", sw.ElapsedMilliseconds, count[0], graphSolutions.Count);
                               //Console.WriteLine("Duped edges: {0}", cg.dupes);
                               Logger.Log(LogLevel.Info, string.Format("Finder completed in {0}ms on {1} edges with {2} solution(s) and {3} dupes", sw.ElapsedMilliseconds, count[0], graphSolutions.Count, cg.dupes));
                           });

                        //h_indexesA = d_indexesA;
                        //h_indexesB = d_indexesB;

                        //var sumA = h_indexesA.Sum(e => e);
                        //var sumB = h_indexesB.Sum(e => e);

                        ;
                    }
                    else
                    {
                        CGraph cg = new CGraph();
                        cg.SetEdges(h_a, (int)count[0]);
                        cg.SetHeader(currentJob);

                        Task.Factory.StartNew(() =>
                        {
                            if (count[0] < 200000)
                            {
                                try
                                {
                                    if (findersInFlight++ < 3)
                                    {
                                        Stopwatch cycleTime = new Stopwatch();
                                        cycleTime.Start();
                                        cg.FindSolutions(graphSolutions);
                                        cycleTime.Stop();
                                        AdjustTrims(cycleTime.ElapsedMilliseconds);
                                        if (graphSolutions.Count > 0)
                                        {
                                            solutions++;
                                        }
                                    }
                                    else
                                        Logger.Log(LogLevel.Warning, "CPU overloaded!");
                                }
                                catch (Exception ex)
                                {
                                    Logger.Log(LogLevel.Error, "Cycle finder crashed: " + ex.Message);
                                }
                                finally
                                {
                                    findersInFlight--;
                                }
                            }
                        });
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log(LogLevel.Error, "Critical error in main cuda loop " + ex.Message);
                    Task.Delay(5000).Wait();
                }
            }

            // clean up
            try
            {
                Task.Delay(500).Wait();

                Comms.Close();

                d_buffer.Dispose();
                d_indexesA.Dispose();
                d_indexesB.Dispose();

                streamPrimary.Dispose();
                streamSecondary.Dispose();

                hAligned_a.Dispose();

                if (ctx != null)
                    ctx.Dispose();
            }
            catch { }
        }

        static void AllocateHostMemory(bool bPinGenericMemory, ref int[] pp_a, ref CudaPageLockedHostMemory<int> pp_Aligned_a, int nbytes)
        {
            //Console.Write("cudaMallocHost() allocating {0:0.00} Mbytes of system memory\n", (float)nbytes / 1048576.0f);
            // allocate host memory (pinned is required for achieve asynchronicity)
            if (pp_Aligned_a != null)
                pp_Aligned_a.Dispose();

            pp_Aligned_a = new CudaPageLockedHostMemory<int>(nbytes / sizeof(int));
            pp_a = new int[nbytes / sizeof(int)];
        }

        private static void AdjustTrims(long elapsedMilliseconds)
        {
            int target = Comms.cycleFinderTargetOverride > 0 ? Comms.cycleFinderTargetOverride :  20 * Environment.ProcessorCount;
            if (elapsedMilliseconds > target)
                trimRounds += 10;
            else
                trimRounds -= 10;

            trimRounds = Math.Max(80, trimRounds);
            trimRounds = Math.Min(300, trimRounds);
        }
    }

    public enum DeviceFamily
    {
        Pascal,
        Turing,
        Other
    }
}
