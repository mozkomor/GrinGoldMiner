
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumeration for the different execution capabilities of devices.
    /// </summary>
    [Flags]
    public enum DeviceExecutionCapabilities : ulong
    {
        /// <summary>
        /// The OpenCL device can execute OpenCL kernels.
        /// </summary>
        Kernel = 1 << 0,

        /// <summary>
        /// The OpenCL device can execute native kernels.
        /// </summary>
        NativeKernel = 1 << 1
    }
}