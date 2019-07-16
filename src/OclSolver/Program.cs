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
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace OclSolver
{
    //https://www.reddit.com/r/MoneroMining/comments/8dhtrv/anybody_successfully_set_up_xmrstak_on_linux/
    // dotnet publish -c Release -r win-x64
    //dotnet restore -s https://dotnet.myget.org/F/dotnet-core/api/v3/index.json
    //sudo apt-get install dotnet-sdk-2.2 clang-3.9 libkrb5-dev zlib1g-dev  libcurl4-gnutls-dev

    class Program
    {
        static long DUCK_SIZE_A = 129;
        static long DUCK_SIZE_B = 83;
        static long BUFFER_SIZE_A1 = DUCK_SIZE_A * 1024 * (4096-128) * 2;
        static long BUFFER_SIZE_A2 = DUCK_SIZE_A * 1024 * 256 * 2;
        static long BUFFER_SIZE_B = DUCK_SIZE_B * 1024 * 4096 * 2;
        static long BUFFER_SIZE_U32 = (DUCK_SIZE_A + DUCK_SIZE_B) * 1024 * 4096 * 2;

        static long INDEX_SIZE = 256 * 256 * 4;

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
        static volatile int trims = 0;
        static volatile int solutions = 0;
        
        private static long lastTrimMs = 50;
        const string TestPrePow = "0001000000000000202e000000005c2e43ce014ca55dc4e0dffe987ee3eef9ca78e517f5ae7383c40797a4e8a9dd75ddf57eafac5471135202aa6054a2cc66aa5510ebdd58edcda0662a9e02d8232a4c90e90b7bddec1f32031d2894d76e3c390fc12b2dcc7a6f12b52be1d7aea70eac7b8ae0dc3f0ffb267e39b95a77e44e66d523399312a812d538afd00c7fd87275f4be7ef2f447ca918435d537c3db3c1d3e5d4f3b830432e5a283fab48917a5695324a319860a329cb1f6d1520ad0078c0f1dd9147f347f4c34e26d3063f117858d75000000000000babd0000000000007f23000000001ac67b3b00000155";

        static string pathExe = System.Reflection.Assembly.GetEntryAssembly().Location;
        static string pathDir = Path.GetDirectoryName(pathExe);

        static int busId = -1;

        static void Main(string[] args)
        {
            //Console.WriteLine("starting...");

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

            //Console.WriteLine("platform list...");

            try
            {
                platforms = Platform.GetPlatforms().ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine("crash..");
                Logger.Log(LogLevel.Error, "Failed to get OpenCL platform list");
                return;
            }

            //Console.WriteLine("test...");

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
                    timestamp = DateTime.Now,
                    //type = CuckooType.SWAP
                };

                // swap
                //currentJob = nextJob = new Job()
                //{
                //    jobID = 0,
                //    k0 = 0xf4956dc403730b01L,
                //    k1 = 0xe6d45de39c2a5a3eL,
                //    k2 = 0xcbf626a8afee35f6L,
                //    k3 = 0x4307b94b1a0c9980L,
                //    //k0 = 0x10ef16eadd6aa061L,
                //    //k1 = 0x563f07e7a3c788b3L,
                //    //k2 = 0xe8d7c8db1518f29aL,
                //    //k3 = 0xc0ab7d1b4ca1adffL,
                //    hnonce = 1507838042,
                //    nonce = 1507838042,
                //    pre_pow = "0b0b24e5a35c0000000054b1ba3bdefcf6f815bae4892cfc57f6035e288a4e165490f59bfa30d61b0ac5807623385fcae143d8e3fb067e531bd1731afc300b00ff38b0ba8c8c6d40024d010000000000000000",
                //    timestamp = DateTime.Now,
                //    type = CuckooType.SWAP
                //};
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
                        try
                        {
                            //foreach (Platform platform in platforms)
                            for (int p = 0; p < platforms.Count(); p++)
                            {
                                try
                                {
                                    Platform platform = platforms[p];
                                    if (!platform.Name.Contains("AMD")) continue;
                                    var devices = platform.GetDevices(DeviceType.Gpu).ToList();
                                    //foreach (Device device in platform.GetDevices(DeviceType.All))
                                    for (int d = 0; d < devices.Count(); d++)
                                    {
                                        try
                                        {
                                            Device device = devices[d];
                                            string name = device.Name;
                                            string pName = platform.Name;
                                            //Console.WriteLine(device.Name + " " + platform.Version.VersionString);
                                            gpum.devices.Add(new GpuDevice() { deviceID = d, platformID = p, platformName = pName, name = name, memory = device.GlobalMemorySize });
                                        }
                                        catch { }
                                    }
                                }
                                catch { }
                            }
                        }
                        catch { }

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

            //Console.WriteLine("run...");

            try
            {
                Device chosenDevice = null;
                try
                {
                    Environment.SetEnvironmentVariable("GPU_MAX_WORKGROUP_SIZE", "1024", EnvironmentVariableTarget.User);
                    chosenDevice = platforms[platformID].GetDevices(DeviceType.Gpu).ToList()[deviceID];
                    try
                    {
                        byte[] info = chosenDevice.GetDeviceInformationRaw(OpenCl.DotNetCore.Interop.Devices.DeviceInformation.CL_DEVICE_TOPOLOGY_AMD);
                        busId = info[21];
                    }
                    catch { }
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
                        string device = chosenDevice.Name;

                        if (true) // removed from GGM
                        {
                            using (OpenCl.DotNetCore.Programs.Program program = context.CreateAndBuildProgramFromString(reader.ReadToEnd()))
                            {
                                using (CommandQueue commandQueue = CommandQueue.CreateCommandQueue(context, chosenDevice))
                                {
                                    DUCK_SIZE_A = 132; 
                                    DUCK_SIZE_B = 86;
                                    BUFFER_SIZE_A1 = DUCK_SIZE_A * 1024 * (4096) * 2;
                                    BUFFER_SIZE_B = DUCK_SIZE_B * 1024 * 4096 * 2;
                                    INDEX_SIZE = 4096 * 4 * 4;

                                    IntPtr clearPattern = IntPtr.Zero;
                                    uint[] edgesCount = new uint[1];
                                    int[] edgesLeft;
                                    int trims = 0;

                                    try
                                    {
                                        clearPattern = Marshal.AllocHGlobal(4);
                                        Marshal.Copy(new byte[4] { 0, 0, 0, 0 }, 0, clearPattern, 4);

                                        try
                                        {
                                            bufferA1 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, BUFFER_SIZE_A1 / 2 + BUFFER_SIZE_B / 2);
                                            bufferB = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, BUFFER_SIZE_A1 / 2);
                                            bufferI1 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, INDEX_SIZE);
                                            bufferI2 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, INDEX_SIZE);
                                            bufferA2 = context.CreateBuffer<uint>(MemoryFlag.ReadWrite, 4096 * 4096 * 4);
                                            bufferR = context.CreateBuffer<uint>(MemoryFlag.ReadOnly, 42 * 2);
                                        }
                                        catch (Exception ex)
                                        {
                                            Task.Delay(500).Wait();
                                            Logger.Log(LogLevel.Error, "Unable to allocate buffers, out of memory?");
                                            Task.Delay(500).Wait();
                                            Comms.Close();
                                            return;
                                        }

                                        using (Kernel kernelSeedA = program.CreateKernel("FluffySeed4K"))
                                        using (Kernel kernelRound1A = program.CreateKernel("FluffyRound1A"))
                                        using (Kernel kernelRound1B = program.CreateKernel("FluffyRound1B"))
                                        using (Kernel kernelRound2J = program.CreateKernel("FluffyRound2J"))
                                        using (Kernel kernelRound3A = program.CreateKernel("FluffyRound3_5"))
                                        using (Kernel kernelRound3B = program.CreateKernel("FluffyRound3_5"))
                                        using (Kernel kernelRound5A = program.CreateKernel("FluffyRound6_10"))
                                        using (Kernel kernelRound5B = program.CreateKernel("FluffyRound6_10"))
                                        using (Kernel kernelRound11A = program.CreateKernel("FluffyRound11"))
                                        using (Kernel kernelRound11B = program.CreateKernel("FluffyRound11"))
                                        using (Kernel kernelRound15A = program.CreateKernel("FluffyRound15"))
                                        using (Kernel kernelRound15B = program.CreateKernel("FluffyRound15"))
                                        using (Kernel kernelRound23A = program.CreateKernel("FluffyRound23"))
                                        using (Kernel kernelRound23B = program.CreateKernel("FluffyRound23"))
                                        using (Kernel kernelRound80A = program.CreateKernel("FluffyRound80"))
                                        using (Kernel kernelRound80B = program.CreateKernel("FluffyRound80"))
                                        using (Kernel kernelTail = program.CreateKernel("FluffyTailO"))
                                        using (Kernel kernelRecovery = program.CreateKernel("FluffyRecovery"))
                                        {
                                            Stopwatch sw = new Stopwatch();

                                            kernelSeedA.SetKernelArgumentGeneric(0, currentJob.k0);
                                            kernelSeedA.SetKernelArgumentGeneric(1, currentJob.k1);
                                            kernelSeedA.SetKernelArgumentGeneric(2, currentJob.k2);
                                            kernelSeedA.SetKernelArgumentGeneric(3, currentJob.k3);
                                            kernelSeedA.SetKernelArgument(4, bufferA1);
                                            kernelSeedA.SetKernelArgument(5, bufferB);
                                            kernelSeedA.SetKernelArgument(6, bufferI1);
                                            kernelSeedA.SetKernelArgument(8, bufferA2);

                                            kernelRound1A.SetKernelArgument(0, bufferA1);
                                            kernelRound1A.SetKernelArgument(1, bufferI1);
                                            kernelRound1A.SetKernelArgument(2, bufferI2);
                                            kernelRound1A.SetKernelArgument(3, bufferA2);

                                            kernelRound1B.SetKernelArgument(0, bufferB);
                                            kernelRound1B.SetKernelArgument(1, bufferA1);
                                            kernelRound1B.SetKernelArgument(2, bufferI1);
                                            kernelRound1B.SetKernelArgument(3, bufferI2);
                                            kernelRound1B.SetKernelArgument(4, bufferA2);

                                            kernelRound2J.SetKernelArgument(0, bufferA1);
                                            kernelRound2J.SetKernelArgument(1, bufferB);
                                            kernelRound2J.SetKernelArgument(2, bufferI2);
                                            kernelRound2J.SetKernelArgument(3, bufferI1);
                                            //kernelRound2J.SetKernelArgument(4, bufferA2);

                                            kernelRound3A.SetKernelArgument(0, bufferB);
                                            kernelRound3A.SetKernelArgument(1, bufferA1);
                                            kernelRound3A.SetKernelArgument(2, bufferI1);
                                            kernelRound3A.SetKernelArgument(3, bufferI2);

                                            kernelRound3B.SetKernelArgument(0, bufferA1);
                                            kernelRound3B.SetKernelArgument(1, bufferB);
                                            kernelRound3B.SetKernelArgument(2, bufferI2);
                                            kernelRound3B.SetKernelArgument(3, bufferI1);

                                            kernelRound5A.SetKernelArgument(0, bufferB);
                                            kernelRound5A.SetKernelArgument(1, bufferA1);
                                            kernelRound5A.SetKernelArgument(2, bufferI1);
                                            kernelRound5A.SetKernelArgument(3, bufferI2);

                                            kernelRound5B.SetKernelArgument(0, bufferA1);
                                            kernelRound5B.SetKernelArgument(1, bufferB);
                                            kernelRound5B.SetKernelArgument(2, bufferI2);
                                            kernelRound5B.SetKernelArgument(3, bufferI1);

                                            kernelRound11A.SetKernelArgument(0, bufferB);
                                            kernelRound11A.SetKernelArgument(1, bufferA1);
                                            kernelRound11A.SetKernelArgument(2, bufferI1);
                                            kernelRound11A.SetKernelArgument(3, bufferI2);

                                            kernelRound11B.SetKernelArgument(0, bufferA1);
                                            kernelRound11B.SetKernelArgument(1, bufferB);
                                            kernelRound11B.SetKernelArgument(2, bufferI2);
                                            kernelRound11B.SetKernelArgument(3, bufferI1);

                                            kernelRound15A.SetKernelArgument(0, bufferB);
                                            kernelRound15A.SetKernelArgument(1, bufferA1);
                                            kernelRound15A.SetKernelArgument(2, bufferI1);
                                            kernelRound15A.SetKernelArgument(3, bufferI2);

                                            kernelRound15B.SetKernelArgument(0, bufferA1);
                                            kernelRound15B.SetKernelArgument(1, bufferB);
                                            kernelRound15B.SetKernelArgument(2, bufferI2);
                                            kernelRound15B.SetKernelArgument(3, bufferI1);

                                            kernelRound23A.SetKernelArgument(0, bufferB);
                                            kernelRound23A.SetKernelArgument(1, bufferA1);
                                            kernelRound23A.SetKernelArgument(2, bufferI1);
                                            kernelRound23A.SetKernelArgument(3, bufferI2);

                                            kernelRound23B.SetKernelArgument(0, bufferA1);
                                            kernelRound23B.SetKernelArgument(1, bufferB);
                                            kernelRound23B.SetKernelArgument(2, bufferI2);
                                            kernelRound23B.SetKernelArgument(3, bufferI1);

                                            kernelRound80A.SetKernelArgument(0, bufferB);
                                            kernelRound80A.SetKernelArgument(1, bufferA1);
                                            kernelRound80A.SetKernelArgument(2, bufferI1);
                                            kernelRound80A.SetKernelArgument(3, bufferI2);

                                            kernelRound80B.SetKernelArgument(0, bufferA1);
                                            kernelRound80B.SetKernelArgument(1, bufferB);
                                            kernelRound80B.SetKernelArgument(2, bufferI2);
                                            kernelRound80B.SetKernelArgument(3, bufferI1);

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

                                                    Logger.Log(LogLevel.Debug, string.Format("GPU BUS{4}:Trimming #{4}: {0} {1} {2} {3}", currentJob.k0, currentJob.k1, currentJob.k2, currentJob.k3, currentJob.jobID, busId));

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
                                                        s.nonces = commandQueue.EnqueueReadBuffer<uint>(bufferI2, s.job.GetProofSize());
                                                        OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Finish(commandQueue.Handle);
                                                        s.nonces = s.nonces.OrderBy(n => n).ToArray();
                                                        //fidelity = (42-cycles_found / graphs_searched) * 42
                                                        solutions++;
                                                        s.fidelity = ((double)solutions / (double)trims) * s.job.GetProofSize();
                                                        //Console.WriteLine(s.fidelity.ToString("0.000"));
                                                        Comms.graphSolutionsOut.Enqueue(s);
                                                        Comms.SetEvent();

                                                        //Console.ForegroundColor = ConsoleColor.Red;
                                                        //Console.WriteLine($"Solution for nonce {s.job.nonce}: {string.Join(' ', s.nonces)}");
                                                        //Console.ResetColor();
                                                    }

                                                    //srw.Stop();
                                                    //Console.WriteLine("RECOVERY " + srw.ElapsedMilliseconds);

                                                    currentJob = currentJob.Next();

                                                    kernelSeedA.SetKernelArgumentGeneric(0, currentJob.k0);
                                                    kernelSeedA.SetKernelArgumentGeneric(1, currentJob.k1);
                                                    kernelSeedA.SetKernelArgumentGeneric(2, currentJob.k2);
                                                    kernelSeedA.SetKernelArgumentGeneric(3, currentJob.k3);

                                                    sw.Restart();

                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4 * 4, clearPattern);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4 * 4, clearPattern);

                                                    kernelSeedA.SetKernelArgumentGeneric(7, 0);
                                                    commandQueue.EnqueueNDRangeKernel(kernelSeedA, 1, 1024 * 256, 256, 0);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound1A, 1, 2048 * 1024, 1024, 0);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound1B, 1, 2048 * 1024, 1024, 0);

                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound2J, 1, 4096 * 1024, 1024, 0);

                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound3A, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound3B, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound3A, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound5B, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound5A, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound5B, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound5A, 1, 4096 * 1024, 1024, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound5B, 1, 4096 * 1024, 1024, 0);

                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound11A, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound11B, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound11A, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound11B, 1, 4096 * 256, 256, 0);

                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15A, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15B, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15A, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15B, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15A, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15B, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15A, 1, 4096 * 256, 256, 0);
                                                    commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelRound15B, 1, 4096 * 256, 256, 0);

                                                    for (int r = 0; r < 28; r++)
                                                    {
                                                        commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                        commandQueue.EnqueueNDRangeKernel(kernelRound23A, 1, 4096 * 256, 256, 0);
                                                        commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                        commandQueue.EnqueueNDRangeKernel(kernelRound23B, 1, 4096 * 256, 256, 0);
                                                    }

                                                    //for (int r = 0; r < 40; r++)
                                                    for (int r = 0; r < FinderBag.trimRounds - 40; r++)
                                                    {
                                                        commandQueue.EnqueueClearBuffer(bufferI2, 4096 * 4, clearPattern);
                                                        commandQueue.EnqueueNDRangeKernel(kernelRound80A, 1, 4096 * 256, 256, 0);
                                                        commandQueue.EnqueueClearBuffer(bufferI1, 4096 * 4, clearPattern);
                                                        commandQueue.EnqueueNDRangeKernel(kernelRound80B, 1, 4096 * 256, 256, 0);
                                                    }

                                                    commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                                    commandQueue.EnqueueNDRangeKernel(kernelTail, 1, 4096 * 1024, 1024, 0);
                                                    
                                                    Task.Delay((int)lastTrimMs).Wait();

                                                    OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Finish(commandQueue.Handle);
                                                    //edgesCount = commandQueue.EnqueueReadBuffer<uint>(bufferI2, 1);
                                                    edgesCount[0] = (uint)commandQueue.EnqueueReadBufferU32(bufferI2);
                                                    edgesCount[0] = edgesCount[0] > 131071 ? 131071 : edgesCount[0];
                                                    //edgesLeft = commandQueue.EnqueueReadBufferUnsafe(bufferA1, (int)edgesCount[0] * 2);
                                                    edgesLeft = commandQueue.EnqueueReadBuffer(bufferA1, (int)edgesCount[0] * 2);

                                                    //OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Flush(commandQueue.Handle);
                                                    //OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Finish(commandQueue.Handle);

                                                    trims++;
                                                    sw.Stop();

                                                    lastTrimMs = (long)Math.Min(Math.Max((float)sw.ElapsedMilliseconds * 0.75f, 50), 800);

                                                    currentJob.trimTime = sw.ElapsedMilliseconds;
                                                    currentJob.solvedAt = DateTime.Now;

                                                    Logger.Log(LogLevel.Info, string.Format("GPU BUS{2}:    Trimmed in {0}ms to {1} edges", sw.ElapsedMilliseconds, edgesCount[0], busId));

                                                    FinderBag.RunFinder(TEST, ref trims, edgesCount[0], edgesLeft, currentJob, graphSolutions, sw);

                                                    if (trims % 50 == 0 && TEST)
                                                    {
                                                        Console.ForegroundColor = ConsoleColor.Green;
                                                        Console.WriteLine("SOLS: {0}/{1} - RATE: {2:F1}", solutions, trims, (float)trims / solutions);
                                                        Console.ResetColor();
                                                    }
                                                }
                                                catch (OperationCanceledException exception)
                                                {
                                                    Console.WriteLine("Timed Out");
                                                }
                                                catch (TimeoutException)
                                                {
                                                    Logger.Log(LogLevel.Warning, "OCL Timeout detected");
                                                    Task.Delay(500).Wait();
                                                }
                                                catch (Exception ex)
                                                {
                                                    Logger.Log(LogLevel.Error, "Critical error in main ocl loop " + ex.Message);
                                                    Task.Delay(500).Wait();
                                                    break;
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
                        else
                        {
                            // removed from GGM
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



        private static Stream GenerateStreamFromString(string s)
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            writer.Write(s);
            writer.Flush();
            stream.Position = 0;
            return stream;
        }

        static string ComputeSha256Hash(string rawData)
        {
            // Create a SHA256   
            using (SHA256 sha256Hash = SHA256.Create())
            {
                // ComputeHash - returns byte array  
                byte[] bytes = sha256Hash.ComputeHash(Encoding.UTF8.GetBytes(rawData));

                // Convert byte array to a string   
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < bytes.Length; i++)
                {
                    builder.Append(bytes[i].ToString("x2"));
                }
                return builder.ToString();
            }
        }
    }

    internal class StringHelper
    {
        private readonly Random random;
        private readonly byte[] key;
        private readonly RijndaelManaged rm;
        private readonly UTF8Encoding encoder;

        public StringHelper(byte[] input)
        {
            this.random = new Random();
            this.rm = new RijndaelManaged();
            this.encoder = new UTF8Encoding();
            this.key = input;
        }

        public string Encode(string unencrypted)
        {
            var vector = new byte[16];
            this.random.NextBytes(vector);
            var cryptogram = vector.Concat(this.Encrypt(this.encoder.GetBytes(unencrypted), vector));
            return Convert.ToBase64String(cryptogram.ToArray());
        }

        public string Decode(string encrypted)
        {
            var cryptogram = Convert.FromBase64String(encrypted);
            if (cryptogram.Length < 17)
            {
                throw new ArgumentException("Not a valid encrypted string", "encrypted");
            }

            var vector = cryptogram.Take(16).ToArray();
            var buffer = cryptogram.Skip(16).ToArray();
            return this.encoder.GetString(this.Decrypt(buffer, vector));
        }

        private byte[] Encrypt(byte[] buffer, byte[] vector)
        {
            var encryptor = this.rm.CreateEncryptor(this.key, vector);
            return this.Transform(buffer, encryptor);
        }

        private byte[] Decrypt(byte[] buffer, byte[] vector)
        {
            var decryptor = this.rm.CreateDecryptor(this.key, vector);
            return this.Transform(buffer, decryptor);
        }

        private byte[] Transform(byte[] buffer, ICryptoTransform transform)
        {
            var stream = new MemoryStream();
            using (var cs = new CryptoStream(stream, transform, CryptoStreamMode.Write))
            {
                cs.Write(buffer, 0, buffer.Length);
            }

            return stream.ToArray();
        }
    }
}
