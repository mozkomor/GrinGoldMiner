
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.CommandQueues
{
    /// <summary>
    /// Represents an enumeration for the command queue properties.
    /// </summary>
    [Flags]
    public enum CommandQueueProperty : ulong
    {
        /// <summary>
        /// Determines whether the commands queued in the command-queue are executed in-order or out-of-order. If set, the commands in the command-queue are executed out-of-order. Otherwise, commands are executed in-order.
        /// </summary>
        OutOfOrderExecutionModeEnable = 1 << 0,

        /// <summary>
        /// Enables or disables profiling of commands in the command-queue. If set, the profiling of commands is enabled. Otherwise profiling of commands is disabled.
        /// </summary>
        ProfilingEnable = 1 << 1,

        /// <summary>
        /// Indicates that the command-queue is a device queue. If <c>CommandQueueProperty.OnDevice</c> is set, <c>CommandQueuePropertyOutOfOrderExecutionModeEnable</c> must also be set. Only out-of-order device queues are supported.
        /// </summary>
        OnDevice = 1 << 2,

        /// <summary>
        /// Indicates that this is the default device queue. This can only be used with <c>CommandQueueProperty.OnDevice</c>. The application must create the default device queue if any kernels containing calls to <see cref="GetDefaultQueue"/>
        /// are enqueued. There can only be one default device queue for each device within a context. <see cref="CreateCommandQueueWithProperties"/> with <c>CommandQueueInformation.Properties</c> set to
        /// <c>CommandQueueProperty.OnDevice | CommandQueueProperty.OnDeviceDefault</c> will return the default device queue that has already been created and increment its retain count by 1.
        /// </summary>
        OnDeviceDefault = 1 << 3
    }
}