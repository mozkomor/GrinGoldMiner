
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Programs
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Programs API.
    /// </summary>
    public static class ProgramsNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Creates a program object for a context, and loads the source code specified by the text strings in the <see cref="strings"/> array into the program object.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="count">The number of source code strings that are provided.</param>
        /// <param name="strings">An array of <see cref="count"/> pointers to optionally null-terminated character strings that make up the source code.</param>
        /// <param name="lengths">
        /// An array with the number of chars in each string (the string length). If an element in <see cref="lengths"/> is zero, its accompanying string is null-terminated. If lengths is <c>null</c>, all strings in the strings argument are
        /// considered null-terminated. Any length value passed in that is greater than zero excludes the null terminator in its count.
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If errorCode is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero program object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the program object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="count"/> is zero or if strings or any entry in strings is <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateProgramWithSource")]
        public static extern IntPtr CreateProgramWithSource(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U4)] uint count,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] strings,
            [In] [MarshalAs(UnmanagedType.LPArray)] uint[] lengths,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates a program object for a context, and loads the binary bits specified by <see cref="binary"/> into the program object.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="numberOfDevices">The number of devices listed in <see cref="deviceList"/>.</param>
        /// <param name="deviceList">A pointer to a list of devices that are in <see cref="context"/>. <see cref="deviceList"/> must be a non-<c>null</c> value. The binaries are loaded for devices specified in this list.</param>
        /// <param name="lengths">An array of the size in bytes of the program binaries to be loaded for devices specified by <see cref="deviceList"/>.</param>
        /// <param name="binaries">
        /// An array of pointers to program binaries to be loaded for devices specified by <see cref="deviceList"/>. For each device given by <c>deviceList[i]</c>, the pointer to the program binary for that device is given by <c>binaries[i]</c>
        /// and the length of this corresponding binary is given by <c>lengths[i]</c>. <c>lengths[i]</c> cannot be zero and <c>binaries[i]</c> cannot be a <c>null</c> pointer.
        /// 
        /// The program binaries specified by <see cref="binaries"/> contain the bits that describe one of the following:
        /// 
        /// - a program executable to be run on the device(s) associated with <see cref="context"/>,
        /// - a compiled program for device(s) associated with <see cref="context"/>, or
        /// - a library of compiled programs for device(s) associated with <see cref="context"/>.
        /// 
        /// The program binary can consist of either or both of device-specific code and/or implementation-specific intermediate representation (IR) which will be converted to the device-specific code.
        /// </param>
        /// <param name="binaryStatus">
        /// Returns whether the program binary for each device specified in <see cref="deviceList"/> was loaded successfully or not. It is an array of <see cref="numberOfDevices"/> entries and returns <c>Result.Success</c> in
        /// <c>binaryStatus[i]</c> if the binary was successfully loaded for the device specified by <c>deviceList[i]</c>. Otherwise returns <c>Result.InvalidValue</c> if <c>lengths[i]</c> is zero or if <c>binaries[i]</c> is a <c>null</c>
        /// value or <c>Result.InvalidBinary</c> in <c>binaryStatus[i]</c> if the program binary is not a valid binary for the specified device. If <see cref="binaryStatus"/> is <c>null</c>, it is ignored.
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero program object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the program object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="deviceList"/> is <c>null</c>, <see cref="numberOfDevices"/> is zero, or if lengths or binaries are <c>null</c> or if any entry in <c>lengths[i]</c> or <c>binaries[i]</c> is <c>null</c>.
        /// 
        /// <c>Result.InvalidDevice</c> if OpenCL devices listed in <see cref="deviceList"/> are not in the list of devices associated with <see cref="context"/>.
        /// 
        /// <c>Result.InvalidBinary</c> if an invalid program binary was encountered for any device. <see cref="binaryStatus"/> will return specific status for each device.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateProgramWithBinary")]
        public static extern IntPtr CreateProgramWithBinary(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfDevices,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] deviceList,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] lengths,
            [In] IntPtr binaries,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result[] binaryStatus,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates a program object for a context, and loads the information related to the built-in kernels into a program object.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="numberOfDevices">The number of devices listed in <see cref="deviceList"/>.</param>
        /// <param name="deviceList">
        /// A pointer to a list of devices that are in <see cref="context"/>. <see cref="deviceList"/> must be a non-<c>null</c> value. The built-in kernels are loaded for devices specified in this list. The devices associated with the program
        /// object will be the list of devices specified by <see cref="deviceList"/>. The list of devices specified by <see cref="deviceList"/> must be devices associated with <see cref="context"/>.
        /// </param>
        /// <param name="kernelNames">A semi-colon separated list of built-in kernel names.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero program object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the program object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="deviceList"/> is <c>null</c>, <see cref="numberOfDevices"/> is zero, or if <see cref="kernelNames"/> is <c>null</c> or <see cref="kernelNames"/> contains a kernel name that is not supported
        /// by any of the devices in <see cref="deviceList"/>.
        /// 
        /// <c>Result.InvalidDevice</c> if OpenCL devices listed in <see cref="deviceList"/> are not in the list of devices associated with <see cref="context"/>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clCreateProgramWithBuiltInKernels")]
        public static extern IntPtr CreateProgramWithBuiltInKernels(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfDevices,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] deviceList,
            [In] [MarshalAs(UnmanagedType.LPStr)] string kernelNames,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates a program object for a context, and loads the IL into the program object.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="il">A pointer to a length-byte block of memory containing SPIR-V or an implementation-defined intermediate language.</param>
        /// <param name="length">The size in bytes of the IL.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero program object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the program object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="il"/> is <c>null</c>, <see cref="length"/> is zero, or if <see cref="length"/>-byte memory pointed to by <see cref="il"/> does not contain well-formed intermediate language input that can be
        /// consumed by the OpenCL runtime.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 1)]
        [DllImport("OpenCL", EntryPoint = "clCreateProgramWithIL")]
        public static extern IntPtr CreateProgramWithIl(
            [In] IntPtr context,
            [In] IntPtr il,
            [In] UIntPtr length,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Increments the program reference count. <see cref="CreateProgram"/> does an implicit retain.
        /// </summary>
        /// <param name="program">The program to retain.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidProgram</c> if <see cref="program"/> is not a valid program object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clRetainProgram")]
        public static extern Result RetainProgram(
            [In] IntPtr program
        );

        /// <summary>
        /// Decrements the program reference count.
        /// </summary>
        /// <param name="program">The program to release.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidProgram</c> if <see cref="program"/> is not a valid program object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clReleaseProgram")]
        public static extern Result ReleaseProgram(
            [In] IntPtr program
        );

        /// <summary>
        /// Builds (compiles and links) a program executable from the program source or binary.
        /// </summary>
        /// <param name="program">The program object.</param>
        /// <param name="numberOfDevices">The number of devices listed in <see cref="deviceList"/>.</param>
        /// <param name="deviceList">
        /// A pointer to a list of devices associated with <see cref="program"/>. If <see cref="deviceList"/> is a <c>null</c> value, the program executable is built for all devices associated with <see cref="program"/> for which a source
        /// or binary has been loaded. If <see cref="deviceList"/> is a non-<c>null</c> value, the program executable is built for devices specified in this list for which a source or binary has been loaded.
        /// </param>
        /// <param name="options">A pointer to a null-terminated string of characters that describes the build options to be used for building the program executable. Certain options are ignored when program is created with IL.</param>
        /// <param name="notificationCallback">
        /// A function pointer to a notification routine. The notification routine is a callback function that an application can register and which will be called when the program executable has been built (successfully or unsuccessfully).
        /// If <see cref="notificationCallback"/> is not <c>null</c>, <see cref="BuildProgram"/> does not need to wait for the build to complete and can return immediately once the build operation can begin. The build operation can begin if
        /// the context, program whose sources are being compiled and linked, list of devices and build options specified are all valid and appropriate host and device resources needed to perform the build are available. If
        /// <see cref="notificationCallback"/> is <c>null</c>, <see cref="BuildProgram"/> does not return until the build has completed. This callback function may be called asynchronously by the OpenCL implementation. It is the
        /// application’s responsibility to ensure that the callback function is thread-safe.
        /// </param>
        /// <param name="userData">Passed as an argument when <see cref="notificationCallback"/> is called. <see cref="userData"/> can be <c>null</c>.</param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clBuildProgram")]
        public static extern Result BuildProgram(
            [In] IntPtr program,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfDevices,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] deviceList,
            [In] [MarshalAs(UnmanagedType.LPStr)] string options,
            [In] IntPtr notificationCallback,
            [In] IntPtr userData
        );

        /// <summary>
        /// Compiles a program’s source for all the devices or a specific device(s) in the OpenCL context associated with <see cref="program"/>.
        /// </summary>
        /// <param name="program">The program object that is the compilation target.</param>
        /// <param name="numberOfDevices">
        /// A pointer to a list of devices associated with <see cref="program"/>. If <see cref="deviceList"/> is a <c>null</c> value, the compile is performed for all devices associated with <see cref="program"/>. If <see cref="deviceList"/>
        /// is a non-<c>null</c> value, the compile is performed for devices specified in this list.
        /// </param>
        /// <param name="deviceList">The number of devices listed in <see cref="deviceList"/>.</param>
        /// <param name="options">A pointer to a null-terminated string of characters that describes the compilation options to be used for building the program executable. Certain options are ignored when program is created with IL.</param>
        /// <param name="numberOfInputHeaders">Specifies the number of programs that describe headers in the array referenced by <see cref="inputHeaders"/>.</param>
        /// <param name="inputHeaders">An array of program embedded headers created with <see cref="CreateProgramWithSource"/>.</param>
        /// <param name="headerIncludeNames">
        /// An array that has a one to one correspondence with <see cref="inputHeaders"/>. Each entry in <see cref="headerIncludeNames"/> specifies the include name used by source in program that comes from an embedded header. The
        /// corresponding entry in <see cref="inputHeaders"/> identifies the program object which contains the header source to be used. The embedded headers are first searched before the headers in the list of directories specified by the
        /// <c>–I</c> compile option. If multiple entries in <see cref="headerIncludeNames"/> refer to the same header name, the first one encountered will be used.
        /// </param>
        /// <param name="notificationCallback">
        /// A function pointer to a notification routine. The notification routine is a callback function that an application can register and which will be called when the program executable has been built (successfully or unsuccessfully).
        /// If <see cref="notificationCallback"/> is not <c>null</c>, <see cref="CompileProgram"/> does not need to wait for the compiler to complete and can return immediately once the compilation can begin. The compilation can begin if the
        /// context, program whose sources are being compiled, list of devices, input headers, programs that describe input headers and compiler options specified are all valid and appropriate host and device resources needed to perform the
        /// compile are available. If <see cref="notificationCallback"/> is <c>null</c>, <see cref="CompileProgram"/> does not return until the compiler has completed. This callback function may be called asynchronously by the OpenCL
        /// implementation. It is the application’s responsibility to ensure that the callback function is thread-safe.
        /// </param>
        /// <param name="userData">Passed as an argument when <see cref="notificationCallback"/> is called. <see cref="userData"/> can be <c>null</c>.</param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clCompileProgram")]
        public static extern Result CompileProgram(
            [In] IntPtr program,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfDevices,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] deviceList,
            [In] [MarshalAs(UnmanagedType.LPStr)] string options,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfInputHeaders,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] inputHeaders,
            [Out] out IntPtr headerIncludeNames,
            [In] IntPtr notificationCallback,
            [In] IntPtr userData
        );

        /// <summary>
        /// Links a set of compiled program objects and libraries for all the devices or a specific device(s) in the OpenCL context and creates a library or executable.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="numberOfDevices">
        /// A pointer to a list of devices associated with <see cref="program"/>. If <see cref="deviceList"/> is a <c>null</c> value, the compile is performed for all devices associated with <see cref="program"/>. If <see cref="deviceList"/>
        /// is a non-<c>null</c> value, the compile is performed for devices specified in this list.
        /// </param>
        /// <param name="deviceList">The number of devices listed in <see cref="deviceList"/>.</param>
        /// <param name="options">A pointer to a null-terminated string of characters that describes the compilation options to be used for building the program executable. Certain options are ignored when program is created with IL.</param>
        /// <param name="numberOfInputPrograms">Specifies the number of programs in array referenced by <see cref="inputPrograms"/>.</param>
        /// <param name="inputPrograms">
        /// An array of program objects that are compiled binaries or libraries that are to be linked to create the program executable. For each device in <see cref="deviceList"/> or if <see cref="deviceList"/> is <c>null</c> the list of
        /// devices associated with context, the following cases occur:
        /// 
        /// - All programs specified by <see cref="inputPrograms"/> contain a compiled binary or library for the device. In this case, a link is performed to generate a program executable for this device.
        /// - None of the programs contain a compiled binary or library for that device. In this case, no link is performed and there will be no program executable generated for this device.
        /// - All other cases will return a <c>Result.InvalidOperation</c> error.
        /// </param>
        /// <param name="notificationCallback">
        /// A function pointer to a notification routine. The notification routine is a callback function that an application can register and which will be called when the program executable has been built (successfully or unsuccessfully).
        /// 
        /// If <see cref="notificationCallback"/> is not <c>null</c>, <see cref="LinkProgram"/> does not need to wait for the linker to complete and can return immediately once the linking operation can begin. Once the linker has completed,
        /// the <see cref="notificationCallback"/> callback function is called which returns the program object returned by <see cref="LinkProgram"/>. The application can query the link status and log for this program object. This callback
        /// function may be called asynchronously by the OpenCL implementation. It is the application’s responsibility to ensure that the callback function is thread-safe.
        /// 
        /// If <see cref="notificationCallback"/> is <c>null</c>, <see cref="LinkProgram"/> does not return until the linker has completed.
        /// </param>
        /// <param name="userData">Passed as an argument when <see cref="notificationCallback"/> is called. <see cref="userData"/> can be <c>null</c>.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clLinkProgram")]
        public static extern IntPtr LinkProgram(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfDevices,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] deviceList,
            [In] [MarshalAs(UnmanagedType.LPStr)] string options,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfInputPrograms,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] inputPrograms,
            [In] IntPtr notificationCallback,
            [In] IntPtr userData,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Allows the implementation to release the resources allocated by the OpenCL compiler for platform.
        /// </summary>
        /// <param name="platform">The platform for which the compiler is to be unloaded.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidPlatform</c> if <see cref="platform"/> is not a valid platform.
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clUnloadPlatformCompiler")]
        public static extern Result UnloadPlatformCompiler(
            [In] IntPtr platform
        );

        /// <summary>
        /// Returns information about the program object.
        /// </summary>
        /// <param name="program">Specifies the program object being queried.</param>
        /// <param name="parameterName">Specifies the information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">The actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not valid, or if size in bytes specified by <see cref="parameterValueSize"/> is less than the size of return type and <see cref="parameterValue"/> is not <c>null</c>.
        /// 
        /// <c>Result.InvalidProgram</c> if <see cref="program"/> is not a valid program object.
        /// 
        /// <c>Result.InvalidProgramExecutable</c> if <see cref="parameterName"/> is <c>ProgramInformation.NumberOfKernels</c> or <c>ProgramInformation.KernelNames</c> and a successful program executable has not been built for at least one
        /// device in the list of devices associated with <see cref="program"/>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetProgramInfo")]
        public static extern Result GetProgramInformation(
            [In] IntPtr program,
            [In] [MarshalAs(UnmanagedType.U4)] ProgramInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Returns build information for each device in the program object.
        /// </summary>
        /// <param name="program">Specifies the program object being queried.</param>
        /// <param name="device">Specifies the device for which build information is being queried. <see cref="device"/> must be a valid device associated with <see cref="program"/>.</param>
        /// <param name="parameterName">Specifies the information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">The actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not a valid device object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/>
        /// is not a <c>null</c> value or if <see cref="parameterName"/> is a value that is available as an extension and the corresponding extension is not supported by the device.
        /// 
        /// <c>Result.InvalidProgram</c> if <see cref="program"/> is not a valid program object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetProgramBuildInfo")]
        public static extern Result GetProgramBuildInformation(
            [In] IntPtr program,
            [In] IntPtr device,
            [In] [MarshalAs(UnmanagedType.U4)] ProgramBuildInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        #endregion

        #region Deprecated Public Methods

        /// <summary>
        /// Allows the implementation to release the resources allocated by the OpenCL compiler.
        /// </summary>
        /// <returns>This call currently always returns <c>Result.Success</c>.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clUnloadCompiler")]
        [Obsolete("This is a deprecated OpenCL 1.1 method, please use UnloadPlatformCompiler instead.")]
        public static extern Result UnloadCompiler();

        #endregion
    }
}