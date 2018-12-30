
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Platforms
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Platforms API.
    /// </summary>
    public static class PlatformsNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Obtain the list of platforms available.
        /// </summary>
        /// <param name="numberOfEntries">The number of platform entries that can be added to <see cref="platforms"/>. If <see cref="platforms"/> is not <c>null</c>, the <see cref="numberOfEntries"/> must be greater than zero.</param>
        /// <param name="platforms">
        /// Returns a list of OpenCL platforms found. The platform values returned in <see cref="platforms"/> can be used to identify a specific OpenCL platform. If <see cref="platforms"/> argument is <c>null</c>, this argument is ignored.
        /// The number of OpenCL platforms returned is the mininum of the value specified by <see cref="numberOfEntries"/> or the number of OpenCL platforms available.
        /// </param>
        /// <param name="numberOfPlatforms">Returns the number of OpenCL platforms available. If <see cref="numberOfPlatforms"/> is <c>null</c>, this argument is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. If the cl_khr_icd extension is enabled, <see cref="GetPlatformIds"/> returns <c>Result.Success</c> if the function is executed successfully and there are a
        /// non zero number of platforms available. Otherwise it returns one of the following errors:
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="numberOfEntries"/> is equal to zero and <see cref="platforms"/> is not <c>null</c>, or if both <see cref="numberOfPlatforms"/> and <see cref="platforms"/> are <c>null</c>.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// 
        /// <c>Result.PlatformNotFoundKhr</c> if the cl_khr_icd extension is enabled and no platforms are found.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetPlatformIDs")]
        public static extern Result GetPlatformIds(
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEntries,
            [Out] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] platforms,
            [Out] [MarshalAs(UnmanagedType.U4)] out uint numberOfPlatforms
        );

        /// <summary>
        /// Get specific information about the OpenCL platform.
        /// </summary>
        /// <param name="platform">The platform ID returned by <see cref="GetPlatformIds"/> or can be <c>null</c>. If <see cref="platform"/> is <c>null</c>, the behavior is implementation-defined.</param>
        /// <param name="parameterName">An enumeration constant that identifies the platform information being queried.</param>
        /// <param name="parameterValueSize">Specifies the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size in bytes must be greater than or equal to size of return type specified above.</param>
        /// <param name="parameterValue">A pointer to memory location where appropriate values for a given <see cref="parameterValue"/> will be returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data being queried by <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following: (The OpenCL specification does not describe the order of precedence for error codes returned by API calls)
        /// 
        /// <c>Result.InvalidPlatform</c> if platform is not a valid platform.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/> is
        /// not a <c>null</c> value.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetPlatformInfo")]
        public static extern Result GetPlatformInformation(
            [In] IntPtr platform,
            [In] [MarshalAs(UnmanagedType.U4)] PlatformInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        #endregion
    }
}