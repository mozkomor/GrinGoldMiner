
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Samplers
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Samplers API.
    /// </summary>
    public static class SamplersNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Creates a sampler object. A sampler object describes how to sample an image when the image is read in the kernel. The built-in functions to read from an image in a kernel take a sampler as an argument. The sampler arguments to
        /// the image read function can be sampler objects created using OpenCL functions and passed as argument values to the kernel or can be samplers declared inside a kernel.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="samplerProperties">
        /// Specifies a list of sampler property names and their corresponding values. Each sampler property name is immediately followed by the corresponding desired value. The list is terminated with 0. If a supported property and its value
        /// is not specified in <see cref="samplerProperties"/>, its default value will be used. <see cref="samplerProperties"/> can be <c>null</c> in which case the default values for supported sampler properties will be used.
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero sampler object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the sampler object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if the property name in <see cref="samplerProperties"/> is not a supported property name, if the value specified for a supported property name is not valid, or if the same property name is specified more
        /// than once.
        /// 
        /// <c>Result.InvalidOperation</c> if <see cref="images"/> are not supported by any device associated with <see cref="context"/>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateSamplerWithProperties")]
        public static extern IntPtr CreateSamplerWithProperties(
            [In] IntPtr context,
            [In] IntPtr samplerProperties,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Increments the sampler reference count.
        /// </summary>
        /// <param name="sampler">Specifies the sampler being retained.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidSampler</c> if <see cref="sampler"/> is not a valid sampler object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clRetainSampler")]
        public static extern Result RetainSample(
            [In] IntPtr sampler
        );

        /// <summary>
        /// Decrements the sampler reference count.
        /// </summary>
        /// <param name="sampler">Specifies the sampler being retained.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidSampler</c> if <see cref="sampler"/> is not a valid sampler object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clReleaseSampler")]
        public static extern Result ReleaseSampler(
            [In] IntPtr sampler
        );

        /// <summary>
        /// Returns information about the sampler object.
        /// </summary>
        /// <param name="sampler">Specifies the sampler being queried.</param>
        /// <param name="parameterName">Specifies the information to query.</param>
        /// <param name="parameterValueSize">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValue">Specifies the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of return type as described in the table above.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not valid, or if size in bytes specified by <see cref="parameterValueSize"/> is less the size of return type as described in the table above and
        /// <see cref="parameterValue"/> is not <c>null</c>.
        /// 
        /// <c>Result.InvalidSampler</c> if <see cref="sampler"/> is not a valid sampler object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetSamplerInfo")]
        public static extern Result GetSamplerInformation(
            [In] IntPtr sampler,
            [In] [MarshalAs(UnmanagedType.U4)] SamplerInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        #endregion

        #region Deprecated Static Methods

        /// <summary>
        /// Creates a sampler object. A sampler object describes how to sample an image when the image is read in the kernel. The built-in functions to read from an image in a kernel take a sampler as an argument. The sampler arguments to
        /// the image read function can be sampler objects created using OpenCL functions and passed as argument values to the kernel or can be samplers declared inside a kernel.
        /// </summary>
        /// <param name="context">Must be a valid OpenCL context.</param>
        /// <param name="normalizedCoordinates">Determines if the image coordinates specified are normalized or not.</param>
        /// <param name="addressingMode">Specifies how out-of-range image coordinates are handled when reading from an image.</param>
        /// <param name="filterMode">Specifies the type of filter that must be applied when reading an image.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero sampler object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the sampler object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following error values
        /// returned in <see cref="errorCode"/>:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if the property name in <see cref="samplerProperties"/> is not a supported property name, if the value specified for a supported property name is not valid, or if the same property name is specified more
        /// than once.
        /// 
        /// <c>Result.InvalidOperation</c> if <see cref="images"/> are not supported by any device associated with <see cref="context"/>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateSampler")]
        [Obsolete("This is a deprecated OpenCL 1.2 method, please use CreateImage instead.")]
        public static extern IntPtr CreateSampler(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U4)] uint normalizedCoordinates,
            [In] [MarshalAs(UnmanagedType.U4)] AddressingMode addressingMode,
            [In] [MarshalAs(UnmanagedType.U4)] FilterMode filterMode,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        #endregion
    }
}