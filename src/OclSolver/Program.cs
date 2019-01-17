using OpenCl.DotNetCore.CommandQueues;
using OpenCl.DotNetCore.Contexts;
using OpenCl.DotNetCore.Devices;
using OpenCl.DotNetCore.Kernels;
using OpenCl.DotNetCore.Memory;
using OpenCl.DotNetCore.Platforms;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using CudaSolver;
using SharedSerialization;
using System.Collections.Concurrent;

namespace OclSolver
{
    //https://www.reddit.com/r/MoneroMining/comments/8dhtrv/anybody_successfully_set_up_xmrstak_on_linux/
    // dotnet publish -c Release -r win-x64
    //dotnet restore -s https://dotnet.myget.org/F/dotnet-core/api/v3/index.json
    //sudo apt-get install dotnet-sdk-2.2 clang-3.9 libkrb5-dev zlib1g-dev  libcurl4-gnutls-dev

    class Program
    {
        const long DUCK_SIZE_A = 129; // AMD 126 + 3
        const long DUCK_SIZE_B = 83;
        const long BUFFER_SIZE_A1 = DUCK_SIZE_A * 1024 * (4096-128) * 2;
        const long BUFFER_SIZE_A2 = DUCK_SIZE_A * 1024 * 256 * 2;
        const long BUFFER_SIZE_B = DUCK_SIZE_B * 1024 * 4096 * 2;
        const long BUFFER_SIZE_U32 = (DUCK_SIZE_A + DUCK_SIZE_B) * 1024 * 4096 * 2;

        const long INDEX_SIZE = 256 * 256 * 4;

        // set this in dry debug runs
        static int platformID = 0;
        static int deviceID = 0;

        static int port = 13500;
        static bool TEST = false;

        static MemoryBuffer bufferA1;
        static MemoryBuffer bufferA2;
        static MemoryBuffer bufferB;
        static MemoryBuffer bufferI1;
        static MemoryBuffer bufferI2;
        static MemoryBuffer bufferR;

        static UInt32[] h_indexesA = new UInt32[INDEX_SIZE];
        static UInt32[] h_indexesB = new UInt32[INDEX_SIZE];

        static Job currentJob;
        static Job nextJob;
        static Stopwatch timer = new Stopwatch();

        public static ConcurrentQueue<Solution> graphSolutions = new ConcurrentQueue<Solution>();
        private volatile static int findersInFlight = 0;
        static volatile int trims = 0;
        static volatile int solutions = 0;
        private static volatile int trimRounds = 80;
        const string TestPrePow = "0001000000000000202e000000005c2e43ce014ca55dc4e0dffe987ee3eef9ca78e517f5ae7383c40797a4e8a9dd75ddf57eafac5471135202aa6054a2cc66aa5510ebdd58edcda0662a9e02d8232a4c90e90b7bddec1f32031d2894d76e3c390fc12b2dcc7a6f12b52be1d7aea70eac7b8ae0dc3f0ffb267e39b95a77e44e66d523399312a812d538afd00c7fd87275f4be7ef2f447ca918435d537c3db3c1d3e5d4f3b830432e5a283fab48917a5695324a319860a329cb1f6d1520ad0078c0f1dd9147f347f4c34e26d3063f117858d75000000000000babd0000000000007f23000000001ac67b3b00000155";

        static void Main(string[] args)
        {
            try
            {
                if (args.Length > 0)
                    deviceID = int.Parse(args[0]); 
                if (args.Length > 2)
                    platformID = int.Parse(args[2]);
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Device ID parse error", ex);
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
                    CGraph.ShowCycles = true;
                    Logger.CopyToConsole = true;
                }
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Master connection error");
            }

            
            // Gets all available platforms and their corresponding devices, and prints them out in a table
            List<Platform> platforms = null;

            try
            {
                platforms = Platform.GetPlatforms().ToList();
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Failed to get OpenCL platform list");
                return;
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
                    //k0 = 0x10ef16eadd6aa061L,
                    //k1 = 0x563f07e7a3c788b3L,
                    //k2 = 0xe8d7c8db1518f29aL,
                    //k3 = 0xc0ab7d1b4ca1adffL,
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
                    Task.Delay(500).Wait();
                    return;
                }

                if (deviceID < 0)
                {
                    try
                    {
                        
                        //Environment.SetEnvironmentVariable("GPU_FORCE_64BIT_PTR", "1", EnvironmentVariableTarget.Machine);
                        Environment.SetEnvironmentVariable("GPU_MAX_HEAP_SIZE", "100", EnvironmentVariableTarget.User);
                        Environment.SetEnvironmentVariable("GPU_USE_SYNC_OBJECTS", "1", EnvironmentVariableTarget.User);
                        Environment.SetEnvironmentVariable("GPU_MAX_ALLOC_PERCENT", "100", EnvironmentVariableTarget.User);
                        Environment.SetEnvironmentVariable("GPU_SINGLE_ALLOC_PERCENT", "100", EnvironmentVariableTarget.User);
                        Environment.SetEnvironmentVariable("GPU_64BIT_ATOMICS", "1", EnvironmentVariableTarget.User);
                        Environment.SetEnvironmentVariable("GPU_MAX_WORKGROUP_SIZE", "1024", EnvironmentVariableTarget.User);
                        //Environment.SetEnvironmentVariable("AMD_OCL_BUILD_OPTIONS_APPEND", "-cl-std=CL2.0", EnvironmentVariableTarget.Machine);

                        GpuDevicesMessage gpum = new GpuDevicesMessage() { devices = new List<GpuDevice>() };
                        //foreach (Platform platform in platforms)
                        for (int p = 0; p < platforms.Count(); p++)
                        {
                            Platform platform = platforms[p];
                            var devices = platform.GetDevices(DeviceType.Gpu).ToList();
                            //foreach (Device device in platform.GetDevices(DeviceType.All))
                            for (int d = 0; d < devices.Count(); d++)
                            {
                                Device device = devices[d];
                                string name = device.Name;
                                string pName = platform.Name;
                                //Console.WriteLine(device.Name + " " + platform.Version.VersionString);
                                gpum.devices.Add(new GpuDevice() { deviceID = d, platformID = p, platformName = pName, name = name, memory = device.GlobalMemorySize });
                            }
                        }
                        Comms.gpuMsg = gpum;
                        Comms.SetEvent();
                        Task.Delay(1000).Wait();
                        Comms.Close();
                        return;
                    }
                    catch (Exception ex)
                    {
                        Logger.Log(LogLevel.Error, "Unable to enumerate OpenCL devices");
                        Task.Delay(500).Wait();
                        Comms.Close();
                        return;
                    }
                }
            }

            try
            {
                Device chosenDevice = null;
                try
                {
                    chosenDevice = platforms[platformID].GetDevices(DeviceType.Gpu).ToList()[deviceID];
                    Console.WriteLine($"Using OpenCL device: {chosenDevice.Name} ({chosenDevice.Vendor})");
                    Console.WriteLine();
                }
                catch (Exception ex)
                {
                    Logger.Log(LogLevel.Error, $"Unable to select OpenCL device {deviceID} on platform {platformID} ");
                    Task.Delay(500).Wait();
                    Comms.Close();
                    return;
                }

                var assembly = Assembly.GetEntryAssembly();
                var resourceStream = assembly.GetManifestResourceStream("OclSolver.kernel.cl");
                using (StreamReader reader = new StreamReader(resourceStream))
                {
                    using (Context context = Context.CreateContext(chosenDevice))
                    {
                        /*
                         * Once the program has been created you can use clGetProgramInfo with CL_PROGRAM_BINARY_SIZES and then CL_PROGRAM_BINARIES, storing the resulting binary programs (one for each device of the context) into a buffer you supply. You can then save this binary data to disk for use in later runs.
                         * Not all devices might support binaries, so you will need to check the CL_PROGRAM_BINARY_SIZES result (it returns a zero size for that device if binaries are not supported).
                         */
                        using (OpenCl.DotNetCore.Programs.Program program = context.CreateAndBuildProgramFromString(reader.ReadToEnd()))
                        {
                            using (CommandQueue commandQueue = CommandQueue.CreateCommandQueue(context, chosenDevice))
                            {
                                IntPtr clearPattern = IntPtr.Zero;
                                uint[] edgesCount;
                                int[] edgesLeft;
                                int trims = 0;
                                try
                                {
                                    clearPattern = Marshal.AllocHGlobal(4);
                                    Marshal.Copy(new byte[4] { 0, 0, 0, 0 }, 0, clearPattern, 4);

                                    try
                                    {
                                        bufferA1 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, BUFFER_SIZE_A1);
                                        bufferA2 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, BUFFER_SIZE_A2);
                                        bufferB = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, BUFFER_SIZE_B);

                                        bufferI1 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, INDEX_SIZE);
                                        bufferI2 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, INDEX_SIZE);

                                        bufferR = context.CreateBuffer<uint>(MemoryFlag.ReadOnly, 42*2);
                                    }
                                    catch (Exception ex)
                                    {
                                        Task.Delay(500).Wait();
                                        Logger.Log(LogLevel.Error, "Unable to allocate buffers, out of memory?");
                                        Task.Delay(500).Wait();
                                        Comms.Close();
                                        return;
                                    }

                                    using (Kernel kernelSeedA = program.CreateKernel("FluffySeed2A"))
                                    using (Kernel kernelSeedB1 = program.CreateKernel("FluffySeed2B"))
                                    using (Kernel kernelSeedB2 = program.CreateKernel("FluffySeed2B"))
                                    using (Kernel kernelRound1 = program.CreateKernel("FluffyRound1"))
                                    using (Kernel kernelRoundO = program.CreateKernel("FluffyRoundNO1"))
                                    using (Kernel kernelRoundNA = program.CreateKernel("FluffyRoundNON"))
                                    using (Kernel kernelRoundNB = program.CreateKernel("FluffyRoundNON"))
                                    using (Kernel kernelTail = program.CreateKernel("FluffyTailO"))
                                    using (Kernel kernelRecovery = program.CreateKernel("FluffyRecovery"))
                                    {
                                        Stopwatch sw = new Stopwatch();

                                        kernelSeedA.SetKernelArgumentGeneric(0, currentJob.k0);
                                        kernelSeedA.SetKernelArgumentGeneric(1, currentJob.k1);
                                        kernelSeedA.SetKernelArgumentGeneric(2, currentJob.k2);
                                        kernelSeedA.SetKernelArgumentGeneric(3, currentJob.k3);
                                        kernelSeedA.SetKernelArgument(4, bufferB);
                                        kernelSeedA.SetKernelArgument(5, bufferA1);
                                        kernelSeedA.SetKernelArgument(6, bufferI1);

                                        kernelSeedB1.SetKernelArgument(0, bufferA1);
                                        kernelSeedB1.SetKernelArgument(1, bufferA1);
                                        kernelSeedB1.SetKernelArgument(2, bufferA2);
                                        kernelSeedB1.SetKernelArgument(3, bufferI1);
                                        kernelSeedB1.SetKernelArgument(4, bufferI2);
                                        kernelSeedB1.SetKernelArgumentGeneric(5, (uint)32);

                                        kernelSeedB2.SetKernelArgument(0, bufferB);
                                        kernelSeedB2.SetKernelArgument(1, bufferA1);
                                        kernelSeedB2.SetKernelArgument(2, bufferA2);
                                        kernelSeedB2.SetKernelArgument(3, bufferI1);
                                        kernelSeedB2.SetKernelArgument(4, bufferI2);
                                        kernelSeedB2.SetKernelArgumentGeneric(5, (uint)0);

                                        kernelRound1.SetKernelArgument(0, bufferA1);
                                        kernelRound1.SetKernelArgument(1, bufferA2);
                                        kernelRound1.SetKernelArgument(2, bufferB);
                                        kernelRound1.SetKernelArgument(3, bufferI2);
                                        kernelRound1.SetKernelArgument(4, bufferI1);
                                        kernelRound1.SetKernelArgumentGeneric(5, (uint)DUCK_SIZE_A * 1024);
                                        kernelRound1.SetKernelArgumentGeneric(6, (uint)DUCK_SIZE_B * 1024);

                                        kernelRoundO.SetKernelArgument(0, bufferB);
                                        kernelRoundO.SetKernelArgument(1, bufferA1);
                                        kernelRoundO.SetKernelArgument(2, bufferI1);
                                        kernelRoundO.SetKernelArgument(3, bufferI2);

                                        kernelRoundNA.SetKernelArgument(0, bufferB);
                                        kernelRoundNA.SetKernelArgument(1, bufferA1);
                                        kernelRoundNA.SetKernelArgument(2, bufferI1);
                                        kernelRoundNA.SetKernelArgument(3, bufferI2);

                                        kernelRoundNB.SetKernelArgument(0, bufferA1);
                                        kernelRoundNB.SetKernelArgument(1, bufferB);
                                        kernelRoundNB.SetKernelArgument(2, bufferI2);
                                        kernelRoundNB.SetKernelArgument(3, bufferI1);

                                        kernelTail.SetKernelArgument(0, bufferB);
                                        kernelTail.SetKernelArgument(1, bufferA1);
                                        kernelTail.SetKernelArgument(2, bufferI1);
                                        kernelTail.SetKernelArgument(3, bufferI2);

                                        kernelRecovery.SetKernelArgumentGeneric(0, currentJob.k0);
                                        kernelRecovery.SetKernelArgumentGeneric(1, currentJob.k1);
                                        kernelRecovery.SetKernelArgumentGeneric(2, currentJob.k2);
                                        kernelRecovery.SetKernelArgumentGeneric(3, currentJob.k3);
                                        kernelRecovery.SetKernelArgument(4, bufferR);
                                        kernelRecovery.SetKernelArgument(5, bufferI2);

                                        int loopCnt = 0;
                                        //for (int i = 0; i < runs; i++)
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

                                                if (!TEST && (currentJob.timestamp.AddMinutes(30) < DateTime.Now) && Comms.lastIncoming.AddMinutes(30) < DateTime.Now)
                                                {
                                                    Logger.Log(LogLevel.Info, string.Format("Job too old..."));
                                                    Task.Delay(1000).Wait();
                                                    continue;
                                                }

                                                // test runs only once
                                                if (TEST && loopCnt++ > 100000)
                                                    Comms.IsTerminated = true;

                                                Logger.Log(LogLevel.Debug, string.Format("GPU AMD{4}:Trimming #{4}: {0} {1} {2} {3}", currentJob.k0, currentJob.k1, currentJob.k2, currentJob.k3, currentJob.jobID, deviceID));

                                                //Stopwatch srw = new Stopwatch();
                                                //srw.Start();

                                                Solution s;
                                                while (graphSolutions.TryDequeue(out s))
                                                {
                                                    kernelRecovery.SetKernelArgumentGeneric(0, s.job.k0);
                                                    kernelRecovery.SetKernelArgumentGeneric(1, s.job.k1);
                                                    kernelRecovery.SetKernelArgumentGeneric(2, s.job.k2);
                                                    kernelRecovery.SetKernelArgumentGeneric(3, s.job.k3);
                                                    commandQueue.EnqueueWriteBufferEdges(bufferR, s.GetLongEdges());
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRecovery, 1, 2048 * 256, 256, 0);
                                                    s.nonces = commandQueue.EnqueueReadBuffer<uint>(bufferI2, 42);
                                                    OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Finish(commandQueue.Handle);
                                                    s.nonces = s.nonces.OrderBy(n => n).ToArray();
                                                    Comms.graphSolutionsOut.Enqueue(s);
                                                    Comms.SetEvent();
                                                }

                                                //srw.Stop();
                                                //Console.WriteLine("RECOVERY " + srw.ElapsedMilliseconds);

                                                currentJob = currentJob.Next();

                                                kernelSeedA.SetKernelArgumentGeneric(0, currentJob.k0);
                                                kernelSeedA.SetKernelArgumentGeneric(1, currentJob.k1);
                                                kernelSeedA.SetKernelArgumentGeneric(2, currentJob.k2);
                                                kernelSeedA.SetKernelArgumentGeneric(3, currentJob.k3);

                                                sw.Restart();

                                                commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                                commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                                commandQueue.EnqueueNDRangeKernel(kernelSeedA, 1, 2048 * 128, 128, 0);
                                                commandQueue.EnqueueNDRangeKernel(kernelSeedB1, 1, 1024 * 128, 128, 0);
                                                commandQueue.EnqueueNDRangeKernel(kernelSeedB2, 1, 1024 * 128, 128, 0);
                                                commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                                commandQueue.EnqueueNDRangeKernel(kernelRound1, 1, 4096 * 1024, 1024, 0);

                                                commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                                commandQueue.EnqueueNDRangeKernel(kernelRoundO, 1, 4096 * 1024, 1024, 0);
                                                commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                                commandQueue.EnqueueNDRangeKernel(kernelRoundNB, 1, 4096 * 1024, 1024, 0);

                                                for (int r = 0; r < trimRounds; r++)
                                                {
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRoundNA, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRoundNB, 1, 4096 * 1024, 1024, 0);
                                                }

                                                commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                                commandQueue.EnqueueNDRangeKernel(kernelTail, 1, 4096 * 1024, 1024, 0);

                                                edgesCount = commandQueue.EnqueueReadBuffer<uint>(bufferI2, 1);
                                                edgesCount[0] = edgesCount[0] > 1000000 ? 1000000 : edgesCount[0];
                                                edgesLeft = commandQueue.EnqueueReadBuffer(bufferA1, (int)edgesCount[0] * 2);

                                                OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Flush(commandQueue.Handle);
                                                OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Finish(commandQueue.Handle);

                                                sw.Stop();

                                                currentJob.trimTime = sw.ElapsedMilliseconds;
                                                currentJob.solvedAt = DateTime.Now;

                                                Logger.Log(LogLevel.Info, string.Format("GPU AMD{2}:    Trimmed in {0}ms to {1} edges", sw.ElapsedMilliseconds, edgesCount[0], deviceID));

                                                CGraph cg = new CGraph();
                                                cg.SetEdges(edgesLeft, (int)edgesCount[0]);
                                                cg.SetHeader(currentJob);

                                                Task.Factory.StartNew(() =>
                                                {
                                                    if (edgesCount[0] < 200000)
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
                                                                if (TEST)
                                                                {
                                                                    Logger.Log(LogLevel.Info, string.Format("Finder completed in {0}ms on {1} edges with {2} solution(s) and {3} dupes", sw.ElapsedMilliseconds, edgesCount[0], graphSolutions.Count, cg.dupes));

                                                                    if (++trims % 50 == 0)
                                                                    {
                                                                        Console.ForegroundColor = ConsoleColor.Green;
                                                                        Console.WriteLine("SOLS: {0}/{1} - RATE: {2:F1}", solutions, trims, (float)trims / solutions);
                                                                        Console.ResetColor();
                                                                    }
                                                                }
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
                                                            Logger.Log(LogLevel.Error, "Cycle finder crashed " + ex.Message);
                                                        }
                                                        finally
                                                        {
                                                            findersInFlight--;
                                                        }
                                                    }
                                                });

                                            }
                                            catch (Exception ex)
                                            {
                                                Logger.Log(LogLevel.Error, "Critical error in main ocl loop " + ex.Message);
                                                Task.Delay(5000).Wait();
                                            }
                                        }

                                        //uint[] resultArray = commandQueue.EnqueueReadBuffer<uint>(bufferI1, 64 * 64);
                                        //uint[] resultArray2 = commandQueue.EnqueueReadBuffer<uint>(bufferI2, 64 * 64);
                                        //Console.WriteLine("SeedA: " + resultArray.Sum(e => e) + " in " + sw.ElapsedMilliseconds / runs);
                                        //Console.WriteLine("SeedB: " + resultArray2.Sum(e => e) + " in " + sw.ElapsedMilliseconds / runs);
                                        //Task.Delay(1000).Wait();
                                        //Console.WriteLine("");
                                    }
                                }
                                finally
                                {
                                    // clear pattern
                                    if (clearPattern != IntPtr.Zero)
                                        Marshal.FreeHGlobal(clearPattern);
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Critical error in OCL Init " + ex.Message);
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(ex.Message);
                Console.ResetColor();
                Task.Delay(500).Wait();
            }
            finally
            {
                Task.Delay(500).Wait();

                try
                {
                    Comms.Close();
                    bufferA1.Dispose();
                    bufferA2.Dispose();
                    bufferB.Dispose();
                    bufferI1.Dispose();
                    bufferI2.Dispose();
                    bufferR.Dispose();

                    if (OpenCl.DotNetCore.CommandQueues.CommandQueue.resultValuePointer != IntPtr.Zero)
                        Marshal.FreeHGlobal(OpenCl.DotNetCore.CommandQueues.CommandQueue.resultValuePointer);
                }
                catch { }
            }

            //Console.ReadKey();
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
}
