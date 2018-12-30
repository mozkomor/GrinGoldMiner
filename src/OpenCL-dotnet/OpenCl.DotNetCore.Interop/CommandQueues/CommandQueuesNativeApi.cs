
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.CommandQueues
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Command Queues API.
    /// </summary>
    public static class CommandQueuesNativeApi
    {
        #region Public Static methods

        /// <summary>
        /// Create a host or device command-queue on a specific device.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="device">
        /// Must be a device or sub-device associated with <see cref="context"/>. It can either be in the list of devices and sub-devices specified when <see cref="context"/> was created using <see cref="CreateContext"/> or be a root device
        /// with the same device type as specified when the context is created using <see cref="CreateContextFromType"/>.
        /// </param>
        /// <param name="properties">
        /// Specifies a list of properties for the command-queue and their corresponding values. Each property name is immediately followed by the corresponding desired value. The list is terminated with 0. If a supported property and its
        /// value is not specified in properties, its default value will be used. <see cref="properties"/> can be <c>null</c> in which case the default values for supported command-queue properties will be used.
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>Returns the created command queue.</returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateCommandQueueWithProperties")]
        public static extern IntPtr CreateCommandQueueWithProperties(
            [In] IntPtr context,
            [In] IntPtr device,
            [In] IntPtr properties,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Increments the command queue reference count.
        /// </summary>
        /// <param name="commandQueue">Specifies the command-queue to retain.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function executed successfully, or one of the errors below:
        /// 
        /// <c>Result.InvalidCommandQueue</c> if <see cref="commandQueue"/> is not a valid command-queue.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clRetainCommandQueue")]
        public static extern Result RetainCommandQueue(
            [In] IntPtr commandQueue
        );

        /// <summary>
        /// Decrements the commandQueue reference count.
        /// </summary>
        /// <param name="commandQueue">Specifies the command-queue to release.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid command queue.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clReleaseCommandQueue")]
        public static extern Result ReleaseCommandQueue(
            [In] IntPtr commandQueue
        );

        /// <summary>
        /// Query information about a command-queue.
        /// </summary>
        /// <param name="commandQueue">Specifies the command-queue being queried.</param>
        /// <param name="parameterName">An enumeration constant that specifies the information to query.</param>
        /// <param name="parameterValueSize">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValue">Specifies the size in bytes of memory pointed to by <see cref="parameterValue"/>.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data being queried by <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function executed successfully, or one of the errors below:
        /// 
        /// <c>Result.InvalidCommandQueue</c> if <see cref="commandQueue"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than the size of the return type <see cref="parameterValue"/>
        /// is not a <c>null</c> value.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetCommandQueueInfo")]
        public static extern Result GetCommandQueueInfo(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] CommandQueueInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Issues all previously queued OpenCL commands in a command-queue to the device associated with the command-queue.
        /// </summary>
        /// <param name="commandQueue">The command queue that is to flushed.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function call was executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidCommandQueue</c> if <see cref="commandQueue"/> is not a valid host command-queue.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clFlush")]
        public static extern Result Flush(
            [In] IntPtr commandQueue
        );

        /// <summary>
        /// Blocks until all previously queued OpenCL commands in a command-queue are issued to the associated device and have completed.
        /// </summary>
        /// <param name="commandQueue">The command queue that is to be finished.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function call was executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidCommandQueue</c> if <see cref="commandQueue"/> is not a valid host command-queue.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clFinish")]
        public static extern Result Finish(
            [In] IntPtr commandQueue
        );

        #endregion

        #region Deprecated Public methods

        /// <summary>
        /// Create a command-queue on a specific device.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="device">
        /// Must be a device associated with <see cref="context"/>. It can either be in the list of devices specified when <see cref="context"/> is created using <see cref="CreateContext"/> or have the same device type as the device type
        /// specified when the context is created using <see cref="CreateContextFromType"/>.
        /// </param>
        /// <param name="properties">
        /// Specifies a list of properties for the command-queue. This is a bit-field described in below. Only command-queue properties specified below can be set in properties; otherwise the value specified in properties is considered to be
        /// not valid.
        /// 
        /// <c>Result.QueueOutOfOrderExecutionModeEnable</c>: Determines whether the commands queued in the command-queue are executed in-order or out-of-order. If set, the commands in the command-queue are executed out-of-order. Otherwise,
        /// commands are executed in-order.
        /// 
        /// <c>Result.QueueProfilingEnable</c>: Enable or disable profiling of commands in the command-queue. If set, the profiling of commands is enabled. Otherwise profiling of commands is disabled.
        /// 
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero command-queue and <see cref="errorCode"/> is set to <c>Result.Success</c> if the command-queue is created successfully. Otherwise, it returns a <c>null</c> value with an error values returned in
        /// <see cref="errorCode"/>.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateCommandQueue")]
        [Obsolete("This is a deprecated OpenCL 1.2 method, please use CreateCommandQueueWithProperties instead.")]
        public static extern IntPtr CreateCommandQueue(
            [In] IntPtr context,
            [In] IntPtr device,
            [In] [MarshalAs(UnmanagedType.U8)] CommandQueueProperty properties,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        #endregion
    }
}