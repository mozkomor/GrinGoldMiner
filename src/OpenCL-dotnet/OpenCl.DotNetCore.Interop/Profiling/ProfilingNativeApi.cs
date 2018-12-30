
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Profiling
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Profiling API.
    /// </summary>
    public static class ProfilingNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Returns profiling information for the command associated with <see cref="event"/> if profiling is enabled.
        /// </summary>
        /// <param name="eventHandle">Specifies the OpenCL event being queried.</param>
        /// <param name="parameterName">An enumeration constant that specifies the information to query.</param>
        /// <param name="parameterValueSize">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValue">Specifies the size in bytes of memory pointed to by <see cref="parameterValue"/>.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data being queried by <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function executed successfully, or one of the errors below:
        /// 
        /// <c>Result.ProfilingInformationNotAvailable</c> if the <c>CommandQueueProperty.ProfilingEnable</c> flag is not set for the command-queue, if the execution status of the command identified by event is not complete or if
        /// <see cref="event"/> refers to the <see cref="EnqueueSvmFree"/> command or is a user event object.
        /// 
        /// <c>Result.InvalidEvent</c> if <see cref="eventHandle"/> is not a valid event.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than the size of the return type <see cref="parameterValue"/>
        /// is not a <c>null</c> value.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetEventProfilingInfo")]
        public static extern Result GetEventProfilingInformation(
            [In] IntPtr eventHandle,
            [In] [MarshalAs(UnmanagedType.U4)] ProfilingInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );
        
        #endregion
    }
}