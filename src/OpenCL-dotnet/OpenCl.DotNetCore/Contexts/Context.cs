
#region Using Directives

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using OpenCl.DotNetCore.Devices;
using OpenCl.DotNetCore.Interop;
using OpenCl.DotNetCore.Interop.Contexts;
using OpenCl.DotNetCore.Interop.Memory;
using OpenCl.DotNetCore.Interop.Programs;
using OpenCl.DotNetCore.Memory;
using OpenCl.DotNetCore.Programs;

#endregion

namespace OpenCl.DotNetCore.Contexts
{
    /// <summary>
    /// Represents an OpenCL context.
    /// </summary>
    public class Context : HandleBase
    {
        #region Constructors

        /// <summary>
        /// Initializes a new <see cref="Context"/> instance.
        /// </summary>
        /// <param name="handle">The handle to the OpenCL context.</param>
        /// <param name="devices">The devices for which the context was created.</param>
        internal Context(IntPtr handle, IEnumerable<Device> devices)
            : base(handle)
        {
            this.Devices = devices;
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets the devices for which the context was created.
        /// </summary>
        public IEnumerable<Device> Devices { get; private set; }

        #endregion

        #region Private Methods

        /// <summary>
        /// Retrieves the specified information about the program build.
        /// </summary>
        /// <typeparam name="T">The type of the data that is to be returned.</param>
        /// <param name="program">The handle to the program for which the build information is to be retrieved.</param>
        /// <param name="device">The device for which the build information is to be retrieved.</param>
        /// <param name="programBuildInformation">The kind of information that is to be retrieved.</param>
        /// <exception cref="OpenClException">If the information could not be retrieved, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the specified information.</returns>
        private T GetProgramBuildInformation<T>(IntPtr program, Device device, ProgramBuildInformation programBuildInformation)
        {
            // Retrieves the size of the return value in bytes, this is used to later get the full information
            UIntPtr returnValueSize;
            Result result = ProgramsNativeApi.GetProgramBuildInformation(program, device.Handle, programBuildInformation, UIntPtr.Zero, null, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The program build information could not be retrieved.", result);
            
            // Allocates enough memory for the return value and retrieves it
            byte[] output = new byte[returnValueSize.ToUInt32()];
            result = ProgramsNativeApi.GetProgramBuildInformation(program, device.Handle, programBuildInformation, new UIntPtr((uint)output.Length), output, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The program build information could not be retrieved.", result);

            // Returns the output
            return InteropConverter.To<T>(output);
        }

        #endregion
        
        #region Public Methods

        /// <summary>
        /// Creates a program from the provided source codes asynchronously. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="sources">The source codes from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Task<Program> CreateAndBuildProgramFromStringAsync(IEnumerable<string> sources)
        {
            // Creates a new task completion source, which is used to signal when the build has completed
            TaskCompletionSource<Program> taskCompletionSource = new TaskCompletionSource<Program>();

            // Loads the program from the specified source string
            Result result;
            IntPtr[] sourceList = sources.Select(source => Marshal.StringToHGlobalAnsi(source)).ToArray();
            uint[] sourceLengths = sources.Select(source => (uint)source.Length).ToArray();
            IntPtr programPointer = ProgramsNativeApi.CreateProgramWithSource(this.Handle, 1, sourceList, sourceLengths, out result);

            // Checks if the program creation was successful, if not, then an exception is thrown
            if (result != Result.Success)
                throw new OpenClException("The program could not be created.", result);

            // Builds (compiles and links) the program and checks if it was successful, if not, then an exception is thrown
            result = ProgramsNativeApi.BuildProgram(programPointer, 0, null, null, Marshal.GetFunctionPointerForDelegate(new BuildProgramCallback((builtProgramPointer, userData) =>
            {
                // Tries to validate the build, if not successful, then an exception is thrown
                try
                {
                    // Cycles over all devices and retrieves the build log for each one, so that the errors that occurred can be added to the exception message (if any error occur during the retrieval, the exception is thrown without the log)
                    Dictionary<string, string> buildLogs = new Dictionary<string, string>();
                    foreach (Device device in this.Devices)
                    {
                        try
                        {
                            string buildLog = this.GetProgramBuildInformation<string>(builtProgramPointer, device, ProgramBuildInformation.Log).Trim();
                            if (!string.IsNullOrWhiteSpace(buildLog))
                                buildLogs.Add(device.Name, buildLog);
                        }
                        catch (OpenClException)
                        {
                            continue;
                        }
                    }

                    // Checks if there were any errors, if so then the build logs are compiled into a formatted string and integrates it into the exception message
                    if (buildLogs.Any())
                    {
                        string buildLogString = string.Join($"{Environment.NewLine}{Environment.NewLine}", buildLogs.Select(keyValuePair => $" Build log for device \"{keyValuePair.Key}\":{Environment.NewLine}{keyValuePair.Value}"));
                        taskCompletionSource.TrySetException(new OpenClException($"The program could not be compiled and linked.{Environment.NewLine}{Environment.NewLine}{buildLogString}", result));
                    }

                    // Since the build was successful, the program is created and the task completion source is resolved with it Creates the new program and returns it
                    taskCompletionSource.TrySetResult(new Program(builtProgramPointer));
                }
                catch (Exception exception)
                {
                    taskCompletionSource.TrySetException(exception);
                }
            })), IntPtr.Zero);

            // Checks if the build could be started successfully, if not, then an exception is thrown
            if (result != Result.Success)
            {
                if (result != Result.Success)
                    taskCompletionSource.TrySetException(new OpenClException("The program could not be compiled and linked.", result));
            }

            // Returns the task which is resolved when the program was build successful or not
            return taskCompletionSource.Task;
        }

        /// <summary>
        /// Creates a program from the provided source codes. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="sources">The source codes from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Program CreateAndBuildProgramFromString(IEnumerable<string> sources)
        {
            // Loads the program from the specified source string
            Result result;
            IntPtr[] sourceList = sources.Select(source => Marshal.StringToHGlobalAnsi(source)).ToArray();
            uint[] sourceLengths = sources.Select(source => (uint)source.Length).ToArray();
            IntPtr programPointer = ProgramsNativeApi.CreateProgramWithSource(this.Handle, 1, sourceList, sourceLengths, out result);

            // Checks if the program creation was successful, if not, then an exception is thrown
            if (result != Result.Success)
                throw new OpenClException("The program could not be created.", result);

            // Builds (compiles and links) the program and checks if it was successful, if not, then an exception is thrown
            result = ProgramsNativeApi.BuildProgram(programPointer, 0, null, "-cl-std=CL2.0", IntPtr.Zero, IntPtr.Zero);
            if (result != Result.Success)
            {
                // Cycles over all devices and retrieves the build log for each one, so that the errors that occurred can be added to the exception message (if any error occur during the retrieval, the exception is thrown without the log)
                Dictionary<string, string> buildLogs = new Dictionary<string, string>();
                foreach (Device device in this.Devices)
                {
                    try
                    {
                        string buildLog = this.GetProgramBuildInformation<string>(programPointer, device, ProgramBuildInformation.Log).Trim();
                        if (!string.IsNullOrWhiteSpace(buildLog))
                            buildLogs.Add(device.Name, buildLog);
                    }
                    catch (OpenClException)
                    {
                        continue;
                    }
                }

                // Compiles the build logs into a formatted string and integrates it into the exception message
                string buildLogString = string.Join($"{Environment.NewLine}{Environment.NewLine}", buildLogs.Select(keyValuePair => $" Build log for device \"{keyValuePair.Key}\":{Environment.NewLine}{keyValuePair.Value}"));
                throw new OpenClException($"The program could not be compiled and linked.{Environment.NewLine}{Environment.NewLine}{buildLogString}", result);
            }

            // Creates the new program and returns it
            return new Program(programPointer);
        }

        /// <summary>
        /// Creates a program from the provided source code asynchronously. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="source">The source code from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Task<Program> CreateAndBuildProgramFromStringAsync(string source) => this.CreateAndBuildProgramFromStringAsync(new List<string> { source });

        /// <summary>
        /// Creates a program from the provided source code. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="source">The source code from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Program CreateAndBuildProgramFromString(string source) => this.CreateAndBuildProgramFromString(new List<string> { source });

        /// <summary>
        /// Creates a program from the provided source streams asynchronously. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="streams">The source streams from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public async Task<Program> CreateAndBuildProgramFromStreamAsync(IEnumerable<Stream> streams)
        {
            // Uses a stream reader to read the all streams
            List<string> sourceList = new List<string>();
            foreach (Stream source in streams)
            {
                using (StreamReader stringReader = new StreamReader(source))
                    sourceList.Add(await stringReader.ReadToEndAsync());
            }

            // Compiles the loaded strings
            return await this.CreateAndBuildProgramFromStringAsync(sourceList);
        }

        /// <summary>
        /// Creates a program from the provided source streams. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="streams">The source streams from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Program CreateAndBuildProgramFromStream(IEnumerable<Stream> streams)
        {
            // Uses a stream reader to read the all streams
            List<string> sourceList = new List<string>();
            foreach (Stream source in streams)
            {
                using (StreamReader stringReader = new StreamReader(source))
                    sourceList.Add(stringReader.ReadToEnd());
            }

            // Compiles the loaded strings
            return this.CreateAndBuildProgramFromString(sourceList);
        }

        /// <summary>
        /// Creates a program from the provided source stream asynchronously. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="stream">The source stream from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Task<Program> CreateAndBuildProgramFromStreamAsync(Stream stream) => this.CreateAndBuildProgramFromStreamAsync(new List<Stream> { stream });

        /// <summary>
        /// Creates a program from the provided source stream. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="stream">The source stream from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Program CreateAndBuildProgramFromStream(Stream stream) => this.CreateAndBuildProgramFromStream(new List<Stream> { stream });

        /// <summary>
        /// Creates a program from the provided source files asynchronously. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="fileNames">The source files from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public async Task<Program> CreateAndBuildProgramFromFileAsync(IEnumerable<string> fileNames)
        {
            // Loads all the source code files and reads them in
            List<string> sourceList = new List<string>();
            foreach (string fileName in fileNames)
            {
                using (StreamReader streamRreader = File.OpenText(fileName))
                    sourceList.Add(await streamRreader.ReadToEndAsync());
            }

            // Compiles and returnes the program
            return await this.CreateAndBuildProgramFromStringAsync(sourceList);
        }

        /// <summary>
        /// Creates a program from the provided source files. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="fileNames">The source files from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Program CreateAndBuildProgramFromFile(IEnumerable<string> fileNames)
        {
            // Loads all the source code files and reads them in
            List<string> sourceList = fileNames.Select(fileName => File.ReadAllText(fileName)).ToList();

            // Compiles and returnes the program
            return this.CreateAndBuildProgramFromString(sourceList);
        }

        /// <summary>
        /// Creates a program from the provided source file asynchronously. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="fileName">The source file from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Task<Program> CreateAndBuildProgramFromFileAsync(string fileName) => this.CreateAndBuildProgramFromFileAsync(new List<string> { fileName });

        /// <summary>
        /// Creates a program from the provided source file. The program is created, compiled, and linked.
        /// </summary>
        /// <param name="fileName">The source file from which the program is to be created.</param>
        /// <exception cref="OpenClException">If the program could not be created, compiled, or linked, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created program.</returns>
        public Program CreateAndBuildProgramFromFile(string fileName) => this.CreateAndBuildProgramFromFile(new List<string> { fileName });

        /// <summary>
        /// Creates a new memory buffer with the specified flags and of the specified size.
        /// </summary>
        /// <param name="memoryFlags">The flags, that determines the how the memory buffer is created and how it can be accessed.</param>
        /// <param name="size">The size of memory that should be allocated for the memory buffer.</param>
        /// <exception cref="OpenClException">If the memory buffer could not be created, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created memory buffer.</returns>
        public MemoryBuffer CreateBuffer(OpenCl.DotNetCore.Memory.MemoryFlag memoryFlags, long size)
        {
            // Creates a new memory buffer of the specified size and with the specified memory flags
            Result result;
            IntPtr memoryBufferPointer = MemoryNativeApi.CreateBuffer(this.Handle, (Interop.Memory.MemoryFlag)memoryFlags, new UIntPtr((ulong)size), IntPtr.Zero, out result);
            
            // Checks if the creation of the memory buffer was successful, if not, then an exception is thrown
            if (result != Result.Success)
                throw new OpenClException("The memory buffer could not be created.", result);

            // Creates the memory buffer from the pointer to the memory buffer and returns it
            return new MemoryBuffer(memoryBufferPointer);
        }

        /// <summary>
        /// Creates a new memory buffer with the specified flags. The size of memory allocated for the memory buffer is determined by <see cref="T"/> and the number of elements.
        /// </summary>
        /// <typeparam name="T">The size of the memory buffer will be determined by the structure specified in the type parameter.</typeparam>
        /// <param name="memoryFlags">The flags, that determines the how the memory buffer is created and how it can be accessed.</param>
        /// <exception cref="OpenClException">If the memory buffer could not be created, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created memory buffer.</returns>
        public MemoryBuffer CreateBuffer<T>(OpenCl.DotNetCore.Memory.MemoryFlag memoryFlags, long size) where T : struct => this.CreateBuffer(memoryFlags, Marshal.SizeOf<T>() * size);

        /// <summary>
        /// Creates a new memory buffer with the specified flags for the specified array. The size of memory 1allocated for the memory buffer is determined by <see cref="T"/> and the number of elements in the array.
        /// </summary>
        /// <typeparam name="T">The size of the memory buffer will be determined by the structure specified in the type parameter.</typeparam>
        /// <param name="memoryFlags">The flags, that determines the how the memory buffer is created and how it can be accessed.</param>
        /// <param name="value">The value that is to be copied over to the device.</param>
        /// <exception cref="OpenClException">If the memory buffer could not be created, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the created memory buffer.</returns>
        public MemoryBuffer CreateBuffer<T>(OpenCl.DotNetCore.Memory.MemoryFlag memoryFlags, T[] value) where T : struct
        {
            // Tries to create the memory buffer, if anything goes wrong, then it is crucial to free the allocated memory
            IntPtr hostBufferPointer = IntPtr.Zero;
            try
            {
                // Determines the size of the specified value and creates a pointer that points to the data inside the structure
                int size = Marshal.SizeOf<T>() * value.Length;
                hostBufferPointer = Marshal.AllocHGlobal(size);
                for (int i = 0; i < value.Length; i++)
                    Marshal.StructureToPtr(value[i], IntPtr.Add(hostBufferPointer, i * Marshal.SizeOf<T>()), false);

                // Creates a new memory buffer for the specified value
                Result result;
                IntPtr memoryBufferPointer = MemoryNativeApi.CreateBuffer(this.Handle, (Interop.Memory.MemoryFlag)memoryFlags, new UIntPtr((uint)size), hostBufferPointer, out result);

                // Checks if the creation of the memory buffer was successful, if not, then an exception is thrown
                if (result != Result.Success)
                    throw new OpenClException("The memory buffer could not be created.", result);

                // Creates the memory buffer from the pointer to the memory buffer and returns it
                return new MemoryBuffer(memoryBufferPointer);
            }
            finally
            {
                // Deallocates the host memory allocated for the value
                if (hostBufferPointer != IntPtr.Zero)
                    Marshal.FreeHGlobal(hostBufferPointer);
            }
        }

        #endregion
        
        #region Public Static Methods

        /// <summary>
        /// Creates a new context for the specified device.
        /// </summary>
        /// <param name="device">The device for which the context is to be created.</param>
        /// <exception cref="OpenClException">If the context could not be created, then an <see cref="OpenClException"/> exception is thrown.</exception>
        /// <returns>Returns the created context.</returns>
        public static Context CreateContext(Device device) => Context.CreateContext(new List<Device> { device });

        /// <summary>
        /// Creates a new context for the specified device.
        /// </summary>
        /// <param name="devices">The devices for which the context is to be created.</param>
        /// <exception cref="OpenClException">If the context could not be created, then an <see cref="OpenClException"/> exception is thrown.</exception>
        /// <returns>Returns the created context.</returns>
        public static Context CreateContext(IEnumerable<Device> devices)
        {
            // Creates the new context for the specified devices
            Result result;
            IntPtr contextPointer = ContextsNativeApi.CreateContext(IntPtr.Zero, (uint)devices.Count(), devices.Select(device => device.Handle).ToArray(), IntPtr.Zero, IntPtr.Zero, out result);

            // Checks if the device creation was successful, if not, then an exception is thrown
            if (result != Result.Success)
                throw new OpenClException("The context could not be created.", result);

            // Creates the new context object from the pointer and returns it
            return new Context(contextPointer, devices);
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Disposes of the resources that have been acquired by the context.
        /// </summary>
        /// <param name="disposing">Determines whether managed object or managed and unmanaged resources should be disposed of.</param>
        protected override void Dispose(bool disposing)
        {
            // Checks if the context has already been disposed of, if not, then the context is disposed of
            if (!this.IsDisposed)
                ContextsNativeApi.ReleaseContext(this.Handle);

            // Makes sure that the base class can execute its dispose logic
            base.Dispose(disposing);
        }

        #endregion

        #region Private Delegates

        /// <summary>
        /// A delegate for the callback of <see cref="BuildProgram"/>.
        /// </summary>
        /// <param name="program">The program that was compiled and linked.</param>
        /// <param name="userData">User-defined data that can be passed to the callback subscription.</param>
        private delegate void BuildProgramCallback(IntPtr program, IntPtr userData);

        #endregion
    }
}