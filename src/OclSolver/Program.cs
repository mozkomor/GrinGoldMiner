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

namespace OclSolver
{
    //https://www.reddit.com/r/MoneroMining/comments/8dhtrv/anybody_successfully_set_up_xmrstak_on_linux/
    // dotnet publish -c Release -r win-x64
    //dotnet restore -s https://dotnet.myget.org/F/dotnet-core/api/v3/index.json
    //sudo apt-get install dotnet-sdk-2.2 clang-3.9 libkrb5-dev zlib1g-dev  libcurl4-gnutls-dev

    class Program
    {
        const long DUCK_SIZE_A = 129; // AMD 126 + 3
        const long DUCK_SIZE_B = 82;
        const long BUFFER_SIZE_A1 = DUCK_SIZE_A * 1024 * (4096-128) * 2;
        const long BUFFER_SIZE_A2 = DUCK_SIZE_A * 1024 * 256 * 2;
        const long BUFFER_SIZE_B = DUCK_SIZE_B * 1024 * 4096 * 2;
        const long BUFFER_SIZE_U32 = (DUCK_SIZE_A + DUCK_SIZE_B) * 1024 * 4096 * 2;

        const long INDEX_SIZE = 256 * 256 * 4;

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

        public static Queue<Solution> graphSolutions = new Queue<Solution>();

        static void Main(string[] args)
        {
            //Environment.SetEnvironmentVariable("GPU_FORCE_64BIT_PTR", "1", EnvironmentVariableTarget.Machine);
            //Environment.SetEnvironmentVariable("GPU_MAX_HEAP_SIZE", "100", EnvironmentVariableTarget.Machine);
            //Environment.SetEnvironmentVariable("GPU_USE_SYNC_OBJECTS", "1", EnvironmentVariableTarget.Machine);
            //Environment.SetEnvironmentVariable("GPU_MAX_ALLOC_PERCENT", "100", EnvironmentVariableTarget.Machine);
            //Environment.SetEnvironmentVariable("GPU_SINGLE_ALLOC_PERCENT", "100", EnvironmentVariableTarget.Machine);
            //Environment.SetEnvironmentVariable("GPU_64BIT_ATOMICS", "1", EnvironmentVariableTarget.Machine);
            //Environment.SetEnvironmentVariable("AMD_OCL_BUILD_OPTIONS_APPEND", "-cl-std=CL2.0", EnvironmentVariableTarget.Machine);
            //var ev =Environment.GetEnvironmentVariables();

            Console.WriteLine("Solver starting!");
            Console.WriteLine();

            try
            {
                if (args.Length > 0)
                    deviceID = int.Parse(args[0]);
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
                    TEST = true;
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Master connection error", ex);
            }

            
            // Gets all available platforms and their corresponding devices, and prints them out in a table
            List<Platform> platforms = null;

            try
            {
                platforms = Platform.GetPlatforms().ToList();
            }
            catch (Exception ex)
            {
                Logger.Log(LogLevel.Error, "Failed to get OpenCL platform list", ex);
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
                    try
                    {
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
                        Logger.Log(LogLevel.Error, "Unable to enumerate OpenCL devices", ex);
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
                    Logger.Log(LogLevel.Error, $"Unable to select OpenCL device {deviceID} on platform {platformID} ", ex);
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
                                    catch { }

                                    using (Kernel kernelSeedA = program.CreateKernel("FluffySeed2A"))
                                    using (Kernel kernelSeedB1 = program.CreateKernel("FluffySeed2B"))
                                    using (Kernel kernelSeedB2 = program.CreateKernel("FluffySeed2B"))
                                    using (Kernel kernelRound1 = program.CreateKernel("FluffyRound1"))
                                    using (Kernel kernelRoundNA = program.CreateKernel("FluffyRoundN"))
                                    using (Kernel kernelRoundNB = program.CreateKernel("FluffyRoundN"))
                                    using (Kernel kernelTail = program.CreateKernel("FluffyTail"))
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

                                        const int runs = 1;

                                        if (runs > 1)
                                        {
                                            commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueNDRangeKernel(kernelSeedA, 1, 2048 * 128, 128, 0);
                                            commandQueue.EnqueueNDRangeKernel(kernelSeedB1, 1, 1024 * 128, 128, 0);
                                            commandQueue.EnqueueNDRangeKernel(kernelSeedB2, 1, 1024 * 128, 128, 0);
                                            commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueNDRangeKernel(kernelRound1, 1, 4096 * 1024, 1024, 0);
                                            commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueNDRangeKernel(kernelRoundNA, 1, 4096 * 1024, 1024, 0);
                                            commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueNDRangeKernel(kernelRoundNB, 1, 4096 * 1024, 1024, 0);
                                            commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueNDRangeKernel(kernelTail, 1, 4096 * 1024, 1024, 0);
                                            OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Finish(commandQueue.Handle);
                                        }
                                        sw.Start();

                                        for (int i = 0; i < runs; i++)
                                        {
                                            commandQueue.EnqueueClearBuffer(bufferI2, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueNDRangeKernel(kernelSeedA, 1, 2048 * 128, 128, 0);
                                            commandQueue.EnqueueNDRangeKernel(kernelSeedB1, 1, 1024 * 128, 128, 0);
                                            commandQueue.EnqueueNDRangeKernel(kernelSeedB2, 1, 1024 * 128, 128, 0);
                                            commandQueue.EnqueueClearBuffer(bufferI1, 64 * 64 * 4, clearPattern);
                                            commandQueue.EnqueueNDRangeKernel(kernelRound1, 1, 4096 * 1024, 1024, 0);

                                            for (int r = 0; r < 80; r++)
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
                                            edgesLeft = commandQueue.EnqueueReadBuffer<int>(bufferA1, (int)edgesCount[0]*2);

                                            OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Flush(commandQueue.Handle);
                                            OpenCl.DotNetCore.Interop.CommandQueues.CommandQueuesNativeApi.Finish(commandQueue.Handle);

                                            CGraph cg = new CGraph();
                                            cg.SetEdges(edgesLeft, (int)edgesCount[0]);
                                            cg.SetHeader(currentJob);

                                            Task.Factory.StartNew(() =>
                                            {
                                                if (edgesCount[0] < 200000)
                                                {
                                                    trims++;
                                                    Queue<Solution> q = new Queue<Solution>();
                                                    try
                                                    {
                                                        cg.FindSolutions(q);
                                                    }
                                                    catch { }
                                                }

                                                //sw.Stop();

                                                ////totalMsCycle += sw.ElapsedMilliseconds;
                                                //Console.WriteLine("Finder completed in {0}ms on {1} edges", sw.ElapsedMilliseconds, edgesCount[0]);
                                                //Console.WriteLine("Duped edges: {0}", cg.dupes);
                                                //Console.WriteLine();
                                            });
                                        }

                                        sw.Stop();

                                        uint[] resultArray = commandQueue.EnqueueReadBuffer<uint>(bufferI1, 64 * 64);
                                        uint[] resultArray2 = commandQueue.EnqueueReadBuffer<uint>(bufferI2, 64 * 64);
                                        Console.WriteLine("SeedA: " + resultArray.Sum(e => e) + " in " + sw.ElapsedMilliseconds / runs);
                                        Console.WriteLine("SeedB: " + resultArray2.Sum(e => e) + " in " + sw.ElapsedMilliseconds / runs);
                                        Task.Delay(1000).Wait();
                                        Console.WriteLine("");
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
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(ex.Message);
                Console.ResetColor();
            }
            finally
            {
                try
                {
                    bufferA1.Dispose();
                    bufferA2.Dispose();
                    bufferB.Dispose();
                    bufferI1.Dispose();
                    bufferI2.Dispose();
                }
                catch { }
            }

            //Console.ReadKey();
        }
    }
}
