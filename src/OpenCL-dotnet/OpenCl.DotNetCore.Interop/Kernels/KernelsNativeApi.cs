
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Kernels
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Kernels API.
    /// </summary>
    public static class KernelsNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Creates a kernel object.
        /// </summary>
        /// <param name="program">A <see cref="program"/> object with a successfully built executable.</param>
        /// <param name="kernelName">A function name in the program declared with the __kernel qualifier.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero kernel object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the kernel object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidProgram</c> if <see cref="program"/> is not a valid program object.
        /// 
        /// <c>Result.InvalidProgramExecutable</c> if there is no successfully built executable for <see cref="program"/>.
        /// 
        /// <c>Result.InvalidKernelName</c> if the function definition for __kernel function given by <see cref="kernelName"/> such as the number of arguments, the argument types are not the same for all devices for which the program
        /// executable has been built.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="kernelName"/> is <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateKernel")]
        public static extern IntPtr CreateKernel(
            [In] IntPtr program,
            [In] [MarshalAs(UnmanagedType.LPStr)] string kernelName,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates kernel objects for all kernel functions in a program object.
        /// </summary>
        /// <param name="program">A program object with a successfully built executable.</param>
        /// <param name="numberOfKernels">The size of memory pointed to by <see cref="kernels"/> specified as the number of kernel entries.</param>
        /// <param name="kernels">
        /// The buffer where the kernel objects for kernels in <see cref="program"/> will be returned. If <see cref="kernels"/> is <c>null</c>, it is ignored. If <see cref="kernels"/> is not <c>null</c>, <see cref="numberOfKernels"/> must be
        /// greater than or equal to the number of kernels in program.
        /// </param>
        /// <param name="numberOfKernelsReturned">The number of kernels in <see cref="program"/>. If <see cref="numberOfKernels"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the kernel objects are successfully allocated. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidProgram</c> if <see cref="program"/> is not a valid program object.
        /// 
        /// <c>Result.InvalidProgramExecutable</c> if there is no successfully built executable for any device in <see cref="program"/>.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="kernels"/> is not <c>null</c> and <see cref="numberOfKernels"/> is less than the number of kernels in program.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateKernelsInProgram")]
        public static extern Result CreateKernelsInProgram(
            [In] IntPtr program,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfKernels,
            [Out] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] kernels,
            [Out] [MarshalAs(UnmanagedType.U4)] out uint numberOfKernelsReturned
        );

        /// <summary>
        /// Make a shallow copy of the kernel object.
        /// </summary>
        /// <param name="sourceKernel">A valid kernel object that will be copied. <see cref="sourceKernel"/> will not be modified in any way by this function.</param>
        /// <param name="errorCode">Assigned an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero kernel object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the kernel is successfully copied. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 1)]
        [DllImport("OpenCL", EntryPoint = "clCloneKernel")]
        public static extern IntPtr CloneKernel(
            [In] IntPtr sourceKernel,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Increments the kernel object reference count. <see cref="CreateKernel"/> or <see cref="CreateKernelsInProgram"/> do an implicit retain.
        /// </summary>
        /// <param name="kernel">Specifies the kernel to retain.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function executed successfully, or one of the errors below:
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clRetainKernel")]
        public static extern Result RetainKernel(
            [In] IntPtr kernel
        );

        /// <summary>
        /// Decrements the kernel reference count.
        /// </summary>
        /// <param name="kernel">The kernel to release.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clReleaseKernel")]
        public static extern Result ReleaseKernel(
            [In] IntPtr kernel
        );

        /// <summary>
        /// Set the argument value for a specific argument of a kernel.
        /// </summary>
        /// <param name="kernel">A valid kernel object.</param>
        /// <param name="argumentIndex">The argument index. Arguments to the kernel are referred by indices that go from 0 for the leftmost argument to n - 1, where n is the total number of arguments declared by a kernel.</param>
        /// <param name="argumentSize">
        /// Specifies the size of the argument value. If the argument is a memory object, the size is the size of the memory object. For arguments declared with the local qualifier, the size specified will be the size in bytes of the buffer
        /// that must be allocated for the local argument. If the argument is of type sampler_t, the <see cref="argumentSize"/> value must be equal to sizeof(cl_sampler). If the argument is of type queue_t, the <see cref="argumentSize"/>
        /// value must be equal to sizeof(cl_commandQueue). For all other arguments, the size will be the size of argument type.
        /// </param>
        /// <param name="argumentValue">
        /// A pointer to data that should be used as the argument value for argument specified by <see cref="argumentIndex"/>. The argument data pointed to by <see cref="argumentValue"/> is copied and the <see cref="argumentValue"/> pointer
        /// can therefore be reused by the application after <see cref="SetKernelArgument"/> returns. The argument value specified is the value used by all API calls that enqueue kernel (<see cref="EnqueueNDRangeKernel"/>) until the argument
        /// value is changed by a call to <see cref="SetKernelArgument"/> for kernel.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clSetKernelArg")]
        public static extern Result SetKernelArgument(
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] uint argumentIndex,
            [In] UIntPtr argumentSize,
            [In] IntPtr argumentValue
        );

        /// <summary>
        /// Set a SVM pointer as the argument value for a specific argument of a kernel.
        /// </summary>
        /// <param name="kernel">A valid kernel object.</param>
        /// <param name="argumentIndex">The argument index. Arguments to the kernel are referred by indices that go from 0 for the leftmost argument to n - 1, where n is the total number of arguments declared by a kernel.</param>
        /// <param name="argumentValue">
        /// The SVM pointer that should be used as the argument value for argument specified by <see cref="argumentIndex"/>. The SVM pointer specified is the value used by all API calls that enqueue kernel (<see cref="EnqueueNDRangeKernel"/>)
        /// until the argument value is changed by a call to <see cref="SetKernelArgumentSvmPointer"/> for <see cref="kernel"/>. The SVM pointer can only be used for arguments that are declared to be a pointer to global or constant memory. The
        /// SVM pointer value must be aligned according to the argument type. For example, if the argument is declared to be global float4 *p, the SVM pointer value passed for p must be at a minimum aligned to a float4. The SVM pointer value
        /// specified as the argument value can be the pointer returned by <see cref="SvmAllocate"/> or can be a pointer + offset into the SVM region.
        /// </param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// 
        /// <c>Result.InvalidArgumentIndex</c> if <see cref="argumentIndex"/> is not a valid argument index.
        /// 
        /// <c>Result.InvalidArgumentValue</c> if <see cref="argumentValue"/> is not a valid value.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clSetKernelArgSVMPointer")]
        public static extern Result SetKernelArgumentSvmPointer(
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] uint argumentIndex,
            [In] IntPtr argumentValue
        );

        /// <summary>
        /// Pass additional information other than argument values to a kernel.
        /// </summary>
        /// <param name="kernel">Specifies the kernel object being queried.</param>
        /// <param name="parameterName">Specifies the information to be passed to <see cref="kernel"/>.</param>
        /// <param name="parameterValueSize">Specifies the size in bytes of the memory pointed to by <see cref="parameterValue"/>.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate values determined by <see cref="parameterName"/> are specified.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not valid, if <see cref="parameterValue"/> is <c>null</c>, or if the size specified by <see cref="parameterValueSize"/> is not valid.
        /// 
        /// <c>Result.InvalidOperation</c> if <see cref="parameterName"/> is <c>KernelExecutionInformation.SvmFineGrainSystem</c> and <see cref="parameterValue"/> is <c>true</c> but no devices in context associated with <see cref="kernel"/>
        /// support fine-grain system SVM allocations.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clSetKernelExecInfo")]
        public static extern Result SetKernelExecutionInformation(
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] KernelExecutionInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [In] IntPtr parameterValue
        );

        /// <summary>
        /// Returns information about the kernel object.
        /// </summary>
        /// <param name="kernel">Specifies the kernel object being queried.</param>
        /// <param name="parameterName">Specifies the information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">The actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/>
        /// is not a <c>null</c> value or if <see cref="parameterName"/> is a value that is available as an extension and the corresponding extension is not supported by the device.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetKernelInfo")]
        public static extern Result GetKernelInformation(
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] KernelInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Returns information about the arguments of a kernel.
        /// </summary>
        /// <param name="kernel">Specifies the kernel object being queried.</param>
        /// <param name="argumentIndex">The argument index. Arguments to the kernel are referred by indices that go from 0 for the leftmost argument to n - 1, where n is the total number of arguments declared by a kernel.</param>
        /// <param name="parameterName">Specifies the argument information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">The actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidArgumentIndex</c> if <see cref="argumentIndex"/> is not a valid argument index.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/>
        /// is not a <c>null</c> value or if <see cref="parameterName"/> is a value that is available as an extension and the corresponding extension is not supported by the device.
        /// 
        /// <c>Result.KernelArgumentInfoNotAvailable</c> if the argument information is not available for <see cref="kernel"/>.
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clGetKernelArgInfo")]
        public static extern Result GetKernelArgumentInformation(
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] uint argumentIndex,
            [In] [MarshalAs(UnmanagedType.U4)] KernelArgumentInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Gets information about the kernel object that may be specific to a device.
        /// </summary>
        /// <param name="kernel">Specifies the kernel object being queried.</param>
        /// <param name="device">
        /// Identifies a specific device in the list of devices associated with <see cref="kernel"/>. The list of devices is the list of devices in the OpenCL context that is associated with <see cref="kernel"/>. If the list of devices
        /// associated with <see cref="kernel"/> is a single device, <see cref="device"/> can be a <c>null</c> value.
        /// </param>
        /// <param name="parameterName">Specifies the argument information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">The actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not in the list of devices associated with <see cref="kernel"/> or if <see cref="device"/> is <c>null</c> but there is more than one device associated with <see cref="kernel"/>.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/>
        /// is not a <c>null</c> value or if <see cref="parameterName"/> is <c>KernelWorkGroupInformation.GlobalWorkSize</c> and <see cref="device"/> is not a custom device and <see cref="kernel"/> is not a built-in kernel.
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetKernelWorkGroupInfo")]
        public static extern Result GetKernelWorkGroupInformation(
            [In] IntPtr kernel,
            [In] IntPtr device,
            [In] [MarshalAs(UnmanagedType.U4)] KernelWorkGroupInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Gets information about the kernel object.
        /// </summary>
        /// <param name="kernel">Specifies the kernel object being queried.</param>
        /// <param name="device">
        /// Identifies a specific device in the list of devices associated with <see cref="kernel"/>. The list of devices is the list of devices in the OpenCL context that is associated with <see cref="kernel"/>. If the list of devices
        /// associated with <see cref="kernel"/> is a single device, <see cref="device"/> can be a <c>null</c> value.
        /// </param>
        /// <param name="parameterName">Specifies the argument information to query.</param>
        /// <param name="inputValueSize">Specifies the size in bytes of memory pointed to by <see cref="inputValue"/>. This size must be equal to the size of the input type.</param>
        /// <param name="inputValue">A pointer to memory where the appropriate parameterization of the query is passed from. If <see cref="inputValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">The actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not in the list of devices associated with <see cref="kernel"/> or if <see cref="device"/> is <c>null</c> but there is more than one device associated with <see cref="kernel"/>.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not valid or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/> is not a <c>null</c> value or
        /// if <see cref="parameterName"/> is <c>KernelSubGroupInformation.SubGroupCountForNDRange</c> and the size in bytes specified by <see cref="inputValueSize"/> is not valid or if <see cref="inputValue"/> is <c>null</c>.
        /// 
        /// <c>Result.InvalidKernel</c> if <see cref="kernel"/> is not a valid kernel object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 1)]
        [DllImport("OpenCL", EntryPoint = "clGetKernelSubGroupInfo")]
        public static extern Result GetKernelSubGroupInformation(
            [In] IntPtr kernel,
            [In] IntPtr device,
            [In] [MarshalAs(UnmanagedType.U4)] KernelSubGroupInformation parameterName,
            [In] UIntPtr inputValueSize,
            [In] IntPtr inputValue,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        #endregion
    }
}