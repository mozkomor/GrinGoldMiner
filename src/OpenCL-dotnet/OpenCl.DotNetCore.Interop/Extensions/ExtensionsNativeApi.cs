
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Extensions
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Extensions API.
    /// </summary>
    public static class ExtensionsNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Gets the address of the extension function named by <see cref="functionName"/> for a given <see cref="platform"/>. Since there is no way to qualify the query with a device, the function pointer returned must work for all
        /// implementations of that extension on different devices for a platform. The behavior of calling a device extension function on a device not supporting that extension is undefined.
        /// </summary>
        /// <param name="platform">The platform for which the extension function is to be retrieved.</param>
        /// <param name="functionName">Name of an extension function.</param>
        /// <returns>
        /// Returns a pointer to the extension function named by <see cref="functionName"/> for a given <see cref="platform"/>. The pointer returned should be cast to a function pointer type matching the extension function's definition defined
        /// in the appropriate extension specification and header file. A return value of <c>null</c> indicates that the specified function does not exist for the implementation or platform is not a valid platform. A non-<c>null</c> return
        /// value for <see cref="GetExtensionFunctionAddressForPlatform"/> does not guarantee that an extension function is actually supported by the platform. The application must also make a corresponding query using
        /// <see cref="GetPlatformInformation"/> or <see cref="GetDeviceInfo"/> to determine if an extension is supported by the OpenCL implementation. <see cref="GetExtensionFunctionAddressForPlatform"/> may not be queried for core
        /// (non-extension) functions in OpenCL. For functions that are queryable with <see cref="GetExtensionFunctionAddressForPlatform"/>, implementations may choose to also export those functions statically from the object libraries
        /// implementing those functions. However, portable applications cannot rely on this behavior. Function pointer type definitions must be declared for all extensions that add API entrypoints. These type definitions are a required part of
        /// the extension interface, to be provided in an appropriate header (such as cl_ext.h if the extension is an OpenCL extension, or cl_gl_ext.h if the extension is an OpenCL/OpenGL sharing extension).
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clGetExtensionFunctionAddressForPlatform")]
        public static extern IntPtr GetExtensionFunctionAddressForPlatform(
            [In] IntPtr platform,
            [In] [MarshalAs(UnmanagedType.LPStr)] string functionName
        );

        #endregion

        #region Deprecated Public Methods

        /// <summary>
        /// Returns the address of the extension function named by <see cref="functionName"/>.
        /// </summary>
        /// <param name="functionName">Name of an extension function.</param>
        /// <returns>
        /// Returns a pointer to the extension function named by <see cref="functionName"/> for a given <see cref="platform"/>. The pointer returned should be cast to a function pointer type matching the extension function's definition defined
        /// in the appropriate extension specification and header file. A return value of <c>null</c> indicates that the specified function does not exist for the implementation or platform is not a valid platform. A non-<c>null</c> return
        /// value for <see cref="GetExtensionFunctionAddress"/> does not guarantee that an extension function is actually supported by the platform. The application must also make a corresponding query using <see cref="GetPlatformInformation"/>
        /// or <see cref="GetDeviceInfo"/> to determine if an extension is supported by the OpenCL implementation. <see cref="GetExtensionFunctionAddress"/> may not be queried for core (non-extension) functions in OpenCL. For functions that are
        /// queryable with <see cref="GetExtensionFunctionAddress"/>, implementations may choose to also export those functions statically from the object libraries implementing those functions. However, portable applications cannot rely on this
        /// behavior. Function pointer type definitions must be declared for all extensions that add API entrypoints. These type definitions are a required part of the extension interface, to be provided in an appropriate header (such as
        /// cl_ext.h if the extension is an OpenCL extension, or cl_gl_ext.h if the extension is an OpenCL/OpenGL sharing extension).
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetExtensionFunctionAddress")]
        [Obsolete("This is a deprecated OpenCL 1.1 method, please use GetExtensionFunctionAddressForPlatform instead.")]
        public static extern IntPtr GetExtensionFunctionAddress(
            [In] [MarshalAs(UnmanagedType.LPStr)] string functionName
        );

        #endregion
    }
}