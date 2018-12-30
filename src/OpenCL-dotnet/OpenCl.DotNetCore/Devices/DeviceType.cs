
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Devices
{
    /// <summary>
    /// Represents the different device types that can be supported by an OpenCL platform.
    /// </summary>
    [Flags]
    public enum DeviceType : ulong
    {
        /// <summary>
        /// The default OpenCL device in the system. The default device cannot be a <c>DeviceType.Custom</c> device.
        /// </summary>
        Default = 1 << 0,

        /// <summary>
        /// An OpenCL device that is the host processor. The host processor runs the OpenCL implementations and is a single or multi-core CPU.
        /// </summary>
        Cpu = 1 << 1,

        /// <summary>
        /// An OpenCL device that is a GPU. By this we mean that the device can also be used to accelerate a 3D API such as OpenGL or DirectX.
        /// </summary>
        Gpu = 1 << 2,

        /// <summary>
        /// Dedicated OpenCL accelerators (for example the IBM CELL Blade). These devices communicate with the host processor using a peripheral interconnect such as PCIe.
        /// </summary>
        Accelerator = 1 << 3,

        /// <summary>
        /// Dedicated accelerators that do not support programs written in OpenCL C.
        /// </summary>
        Custom = 1 << 4,

        /// <summary>
        /// All OpenCL devices available in the system except <c>DeviceType.Custom</c> devices.
        /// </summary>
        All = 0xFFFFFFFF
    }
}