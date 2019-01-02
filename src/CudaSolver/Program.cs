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

// dotnet publish -c Release -r win-x64

namespace CudaSolver
{
    class Program
    {
        const long DUCK_SIZE_A = 129L;
        const long DUCK_SIZE_B = 82L;
        const long BUFFER_SIZE_A = DUCK_SIZE_A * 1024 * 4096;
        const long BUFFER_SIZE_B = DUCK_SIZE_B * 1024 * 4096;
        const long BUFFER_SIZE_U32 = (DUCK_SIZE_A + DUCK_SIZE_B) * 1024 * 4096;
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
        static CudaKernel meanRound;
        static CudaKernel meanTail;
        static CudaKernel meanRecover;

        static CudaDeviceVariable<ulong> d_buffer;
        static CudaDeviceVariable<ulong> d_bufferMid;
        static CudaDeviceVariable<ulong> d_bufferB;
        static CudaDeviceVariable<UInt32> d_indexesA;
        static CudaDeviceVariable<UInt32> d_indexesB;

        static CudaStream streamPrimary;
        static CudaStream streamSecondary;
        static int[] h_a = null;
        static CudaPageLockedHostMemory<int> hAligned_a = null;

        static UInt32[] h_indexesA = new UInt32[INDEX_SIZE];
        static UInt32[] h_indexesB = new UInt32[INDEX_SIZE];

        static Job currentJob;
        static Job nextJob;
        static Stopwatch timer = new Stopwatch();

        public static Queue<Solution> graphSolutions = new Queue<Solution>();

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
                    TEST = true;
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Port parse error");
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
                    pre_pow = "0001000000000000100f000000005c1fea7f0208c1ae873960d0f98e0d3b837fc9a08b898d8b2f5d067f98e74b7f0cedeed42b1158649fc4638f5bc548f6296f57e09966c0968be97780b8a842957c329b2291c8e3dcf52d2558586a6eaecb416693567fe1841b2a9375ff448b7003de59752678e01774981e487a7aec9f198d5dba2acc4a50cac9b4d7f0b82768ed26721cabbe6df16becaed640c169289e7a66a15641a3b1a584aa2d192b9bdfbac99f6d86f43075896fa93cb50b85dbbb1405059e5402c5eeb1614c71c48f81d3add3520000000000002ee40000000000002770000000001cb576280000028f"
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
                        gpum.devices.Add(new GpuDevice() {id = i, name = name, memory = info.TotalGlobalMemory });
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

                ctx = new CudaContext(deviceID, CUCtxFlags.BlockingSync | CUCtxFlags.MapHost);
                meanSeedA = ctx.LoadKernelPTX(resourceStream, "FluffySeed2A");
                meanSeedA.BlockDimensions = 128;
                meanSeedA.GridDimensions = 2048;
                meanSeedA.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanSeedB = ctx.LoadKernelPTX(resourceStream, "FluffySeed2B");
                meanSeedB.BlockDimensions = 128;
                meanSeedB.GridDimensions = 1024;
                meanSeedB.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

                meanRound = ctx.LoadKernelPTX(resourceStream, "FluffyRound");
                meanRound.BlockDimensions = 512;
                meanRound.GridDimensions = 4096;
                meanRound.PreferredSharedMemoryCarveout = CUshared_carveout.MaxShared;

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
                Logger.Log(LogLevel.Error, "Unable to create kernels", ex);
            }

            try
            {
                d_buffer = new CudaDeviceVariable<ulong>(BUFFER_SIZE_U32);
                d_bufferMid = new CudaDeviceVariable<ulong>(d_buffer.DevicePointer + (BUFFER_SIZE_B * 8));
                d_bufferB = new CudaDeviceVariable<ulong>(d_buffer.DevicePointer + (BUFFER_SIZE_A * 8));

                d_indexesA = new CudaDeviceVariable<uint>(INDEX_SIZE);
                d_indexesB = new CudaDeviceVariable<uint>(INDEX_SIZE);

                Array.Clear(h_indexesA, 0, h_indexesA.Length);
                Array.Clear(h_indexesB, 0, h_indexesA.Length);

                d_indexesA = h_indexesA;
                d_indexesB = h_indexesB;

                streamPrimary = new CudaStream(CUStreamFlags.NonBlocking);
                streamSecondary = new CudaStream(CUStreamFlags.NonBlocking);
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Unable to create buffers, out of memory?", ex);
            }

            try
            {
                AllocateHostMemory(true, ref h_a, ref hAligned_a, 1024*1024*32);
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Unable to create pinned memory.", ex);
            }

            int loopCnt = 0;

            while (!Comms.IsTerminated)
            {
                try
                {
                    if (Comms.nextJob.pre_pow == null || Comms.nextJob.pre_pow == "")
                    {
                        Logger.Log(LogLevel.Info, string.Format("Waiting for job...."));
                        Task.Delay(1000).Wait();
                        continue;
                    }

                    // test runs only once
                    if (TEST && loopCnt++ > 100)
                        Comms.IsTerminated = true;

                    if (!TEST && (currentJob.pre_pow != Comms.nextJob.pre_pow))
                    {
                        currentJob = Comms.nextJob;
                     }
                    currentJob.MutateJob();

                    Logger.Log(LogLevel.Info, string.Format("Trimming #{4}: {0} {1} {2} {3}", currentJob.k0, currentJob.k1, currentJob.k2, currentJob.k3, currentJob.jobID));

                    if (graphSolutions.Count > 0)
                    {
                        Solution s;
                        lock (graphSolutions)
                        {
                            s = graphSolutions.Dequeue();
                        }
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

                    timer.Restart();

                    d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);

                    meanSeedA.RunAsync(streamPrimary.Stream, currentJob.k0, currentJob.k1, currentJob.k2, currentJob.k3, d_bufferMid.DevicePointer, d_indexesB.DevicePointer);
                    meanSeedB.RunAsync(streamPrimary.Stream, d_bufferMid.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, 0);
                    meanSeedB.RunAsync(streamPrimary.Stream, d_bufferMid.DevicePointer, d_buffer.DevicePointer + ((BUFFER_SIZE_A * 8) / 2), d_indexesB.DevicePointer, d_indexesA.DevicePointer, 32);

                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer, d_bufferB.DevicePointer, d_indexesA.DevicePointer, d_indexesB.DevicePointer, DUCK_EDGES_A, DUCK_EDGES_B);

                    d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_bufferB.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, DUCK_EDGES_B, DUCK_EDGES_B / 2);
                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer, d_bufferB.DevicePointer, d_indexesA.DevicePointer, d_indexesB.DevicePointer, DUCK_EDGES_B / 2, DUCK_EDGES_B / 2);
                    d_indexesA.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_bufferB.DevicePointer, d_buffer.DevicePointer, d_indexesB.DevicePointer, d_indexesA.DevicePointer, DUCK_EDGES_B / 2, DUCK_EDGES_B / 2);
                    d_indexesB.MemsetAsync(0, streamPrimary.Stream);
                    meanRound.RunAsync(streamPrimary.Stream, d_buffer.DevicePointer, d_bufferB.DevicePointer, d_indexesA.DevicePointer, d_indexesB.DevicePointer, DUCK_EDGES_B / 2, DUCK_EDGES_B / 4);

                    for (int i = 0; i < 80; i++)
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
                    currentJob.trimTime = timer.ElapsedMilliseconds;

                    //Console.WriteLine("Trimmed in {0}ms to {1} edges", timer.ElapsedMilliseconds, count[0]);
                    Logger.Log(LogLevel.Info, string.Format("Trimmed in {0}ms to {1} edges", timer.ElapsedMilliseconds, count[0]));

                    if (TEST)
                    {
                        //Console.WriteLine("Trimmed in {0}ms to {1} edges", timer.ElapsedMilliseconds, count[0]);

                        CGraph cg = new CGraph();
                        cg.SetEdges(h_a, (int)count[0]);
                        cg.SetHeader(currentJob);

                        currentJob = currentJob.Next();

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
                                           cg.FindSolutions(graphSolutions);
                                           if (graphSolutions.Count > 0) solutions++;
                                       }
                                       else
                                           Logger.Log(LogLevel.Warning, "CPU overloaded!");
                                   }
                                   catch (Exception ex)
                                   {
                                       Logger.Log(LogLevel.Error, "Cycle finder error", ex);
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
                                   Console.WriteLine("LOSS: {0}/{1}", solutions, trims);
                                   Console.ResetColor();
                               }
                               //Console.WriteLine("Finder completed in {0}ms on {1} edges with {2} solution(s)", sw.ElapsedMilliseconds, count[0], graphSolutions.Count);
                               //Console.WriteLine("Duped edges: {0}", cg.dupes);
                               Logger.Log(LogLevel.Info, string.Format("Finder completed in {0}ms on {1} edges with {2} solution(s) and {3} dupes", sw.ElapsedMilliseconds, count[0], graphSolutions.Count, cg.dupes));
                               //Console.WriteLine();
                               graphSolutions.Clear();
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
                                        cg.FindSolutions(graphSolutions);
                                        if (graphSolutions.Count > 0) solutions++;
                                    }
                                    else
                                        Logger.Log(LogLevel.Warning, "CPU overloaded!");
                                }
                                catch
                                { }
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
                    Logger.Log(LogLevel.Error, "Critical error in main loop", ex);
                    Task.Delay(5000).Wait();
                }
            }

            // clean up
            try
            {
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

            Task.Delay(500).Wait();
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
    }

    public enum DeviceFamily
    {
        Pascal,
        Turing,
        Other
    }
}
