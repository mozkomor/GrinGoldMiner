
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Devices API.
    /// </summary>
    public static class DevicesNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Obtain the list of devices available on a platform.
        /// </summary>
        /// <param name="platform">Refers to the platform ID returned by <see cref="GetPlatformIds"/< or can be <c>null</c>. If <see cref="platform"/> is <c>null</c>, the behavior is implementation-defined.</param>
        /// <param name="deviceType">A bitfield that identifies the type of OpenCL device. The <see cref="deviceType"/> can be used to query specific OpenCL devices or all OpenCL devices available.</param>
        /// <param name="numberOfEntries">The number of device entries that can be added to <see cref="devices"/>. If <see cref="devices"/> is not <c>null</c>, the <see cref="numberOfEntries"/> must be greater than zero.</param>
        /// <param name="devices">
        /// A list of OpenCL devices found. The device values returned in <see cref="devices"/> can be used to identify a specific OpenCL device. If <see cref="devices"/> argument is <c>null</c>, this argument is ignored. The number of
        /// OpenCL devices returned is the mininum of the value specified by <see cref="numberOfEntries"/> or the number of OpenCL devices whose type matches <see cref="deviceType"/>.
        /// </param>
        /// <param name="numberOfDevicesReturned">The number of OpenCL devices available that match <see cref="deviceType". If <see cref="numberOfDevicesReturned"/> is <c>null</c>, this argument is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise it returns one of the following errors:
        /// 
        /// <c>Result.InvalidPlatform</c> if <see cref="platform"/> is not a valid platform.
        /// 
        /// <c>Result.InvalidDeviceType</c> if <see cref="deviceType"/> is not a valid value.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="numberOfEntries"/> is equal to zero and <see cref="devices"/> is not <c>null</c> or if both <see cref="numberOfDevicesReturned"/> and <see cref="devices"/> are <c>null</c>.
        /// 
        /// <c>Result.DeviceNotFound</c> if no OpenCL devices that matched <see cref="deviceType"/> were found.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetDeviceIDs")]
        public static extern Result GetDeviceIds(
            [In] IntPtr platform,
            [In] [MarshalAs(UnmanagedType.U8)] DeviceType deviceType,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEntries,
            [Out] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] devices,
            [Out] [MarshalAs(UnmanagedType.U4)] out uint numberOfDevicesReturned
        );

        /// <summary>
        /// Get information about an OpenCL device.
        /// </summary>
        /// <param name="device">
        /// A device returned by <see cref="GetDeviceIds"/>. May be a device returned by <see cref="GetDeviceIds"/> or a sub-device created by <see cref="CreateSubDevices"/>. If device is a sub-device, the specific information for the
        /// sub-device will be returned.
        /// </param>
        /// <param name="parameterName">An enumeration constant that identifies the device information being queried.</param>
        /// <param name="parameterValueSize">Specifies the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size in bytes must be greater than or equal to the size of return type specified.</param>
        /// <param name="parameterValue">A pointer to memory location where appropriate values for a given <see cref="parameterName"/>. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data being queried by <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not valid.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/> is
        /// not a <c>null</c> value or if <see cref="parameterName"/> is a value that is available as an extension and the corresponding extension is not supported by the device.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetDeviceInfo")]
        public static extern Result GetDeviceInformation(
            [In] IntPtr device,
            [In] [MarshalAs(UnmanagedType.U4)] DeviceInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Creates an array of sub-devices that each reference a non-intersecting set of compute units within <see cref="inDevice"/>.
        /// </summary>
        /// <param name="inDevice">The device to be partitioned.</param>
        /// <param name="properties">
        /// Specifies how <see cref="inDevice"/> is to be partitioned, described by a partition name and its corresponding value. Each partition name is immediately followed by the corresponding desired value. The list is terminated with 0.
        /// </param>
        /// <param name="numberOfDevices">Size of memory pointed to by <see cref="outDevices"/> specified as the number of device entries.</param>
        /// <param name="outDevices"></param>
        /// <param name="numberOfDevicesReturned">
        /// The buffer where the OpenCL sub-devices will be returned. If <see cref="outDevices"/> is <c>null</c>, this argument is ignored. If <see cref="outDevices"/> is not <c>null</c>, <see cref="numberOfDevices"/> must be greater than
        /// or equal to the number of sub-devices that device may be partitioned into according to the partitioning scheme specified in properties.
        /// </param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not valid.
        /// 
        /// <c>Result.InvalidValue</c> if values specified in <see cref="properties"/> are not valid or if values specified in <see cref="properties"/> are valid but not supported by the device. If <see cref="outDevices"/> is not <c>null</c> and
        /// <see cref="numberOfDevices"/> is less than the number of sub-devices created by the partition scheme.
        /// 
        /// <c>Result.DevicePartitionFailed</c> if the partition name is supported by the implementation but <see cref="inDevice"/> could not be further partitioned.
        /// 
        /// <c>Result.InvalidDevicePartitionCount</c> if the partition name specified in properties is <c>DevicePartitionProperty.PartitionByCounts</c> and the number of sub-devices requested exceeds the maximum number of sub-devices or the total
        /// number of compute units requested exceeds the maximum number of compute units for <see cref="inDevice"/>, or the number of compute units requested for one or more sub-devices is less than zero or the number of sub-devices requested
        /// exceeds the maximum number of compute units for <see cref="inDevice"/>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clCreateSubDevices")]
        public static extern Result CreateSubDevices(
            [In] IntPtr inDevice,
            [In] IntPtr properties,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfDevices,
            [Out] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] outDevices,
            [Out] [MarshalAs(UnmanagedType.U4)] out uint numberOfDevicesReturned
        );

        /// <summary>
        /// Increments the device reference count.
        /// </summary>
        /// <param name="device">Specifies the device to retain.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function executed successfully, or one of the errors below:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not a valid command-queue.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clRetainDevice")]
        public static extern Result RetainDevice(
            [In] IntPtr device
        );

        /// <summary>
        /// Decrements the device reference count if device is a valid sub-device created by a call to <see cref="CreateSubDevices"/>. If device is a root level device i.e. a device returned by <see cref="GetDeviceIDs"/>, the device
        /// reference count remains unchanged.
        /// </summary>
        /// <param name="device">The device to release.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not a valid device object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clReleaseDevice")]
        public static extern Result ReleaseDevice(
            [In] IntPtr device
        );

        /// <summary>
        /// Replaces the default command queue on the device. <see cref="SetDefaultDeviceCommandQueue"/> may be used to replace a default device command queue created with <see cref="CreateCommandQueueWithProperties"/> and the
        /// <c>CommandQueueProperty.OnDeviceDefault</c> flag.
        /// </summary>
        /// <param name="context">The contact to which the device belongs.</param>
        /// <param name="device">The device whose default device command queue is to be replaced.</param>
        /// <param name="commandQueue">The command queue that is to be replace the current default command queue as default command queue for the device.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not a valid device object.
        /// 
        /// <c>Result.InvalidCommandQueue</c> if <see cref="commandQueue"/> is not a valid command queue for <see cref="device"/>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 1)]
        [DllImport("OpenCL", EntryPoint = "clSetDefaultDeviceCommandQueue")]
        public static extern Result SetDefaultDeviceCommandQueue(
            [In] IntPtr context,
            [In] IntPtr device,
            [In] IntPtr commandQueue
        );

        /// <summary>
        /// Returns a reasonably synchronized pair of timestamps from the device timer and the host timer as seen by device.
        /// </summary>
        /// <param name="device">A device returned by <see cref="GetDeviceIds"/>.</param>
        /// <param name="deviceTimestamp">
        /// Will be updated with the value of the current timer in nanoseconds. The resolution of the timer is the same as the device profiling timer returned by <see cref="GetDeviceInformation"/> and the
        /// <c>DeviceInformation.ProfilingTimerResolution</c> query.
        /// </param>
        /// <param name="hostTimestamp">
        /// Will be updated with the value of the current timer in nanoseconds at the closest possible point in time to that at which device timer was returned. The resolution of the timer may be queried via <see cref="GetPlatformInfo"/>
        /// and the flag <c>PlatformInformation.HostTimerResolution</c>.
        /// </param>
        /// <returns>
        /// Returns <c>Result.Success</c> with a time value in <see cref="hostTimestamp"/> if provided. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not a valid OpenCL device.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="hostTimestamp"/> or <see cref="deviceTimestamp"/> is <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 1)]
        [DllImport("OpenCL", EntryPoint = "clGetDeviceAndHostTimer")]
        public static extern Result GetDeviceAndHostTimer(
            [In] IntPtr device,
            [In] IntPtr deviceTimestamp,
            [In] IntPtr hostTimestamp
        );

        /// <summary>
        /// Return the current value of the host clock as seen by device.
        /// </summary>
        /// <param name="device">A device returned by <see cref="GetDeviceIds"/>.</param>
        /// <param name="hostTimestamp">
        /// Will be updated with the value of the current timer in nanoseconds at the closest possible point in time to that at which device timer was returned. The resolution of the timer may be queried via <see cref="GetPlatformInfo"/>
        /// and the flag <c>PlatformInformation.HostTimerResolution</c>.
        /// </param>
        /// <returns>
        /// Returns <c>Result.Success</c> with a time value in <see cref="hostTimestamp"/> if provided. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidDevice</c> if <see cref="device"/> is not a valid OpenCL device.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="hostTimestamp"/> is <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 1)]
        [DllImport("OpenCL", EntryPoint = "clGetHostTimer")]
        public static extern Result GetHostTimer(
            [In] IntPtr device,
            [In] IntPtr hostTimestamp
        );

        #endregion
    }
}