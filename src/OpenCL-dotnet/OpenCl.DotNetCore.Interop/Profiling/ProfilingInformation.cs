
namespace OpenCl.DotNetCore.Interop
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried for profiling purposes.
    /// </summary>
    public enum ProfilingInformation : uint
    {
        /// <summary>
        /// A 64-bit value that describes the current device time counter in nanoseconds when the command identified by the event is enqueued in a command-queue by the host.
        /// </summary>
        CommandQueued = 0x1280,

        /// <summary>
        /// A 64-bit value that describes the current device time counter in nanoseconds when the command identified by the event that has been enqueued is submitted by the host to the device associated with the command-queue.
        /// </summary>
        CommandSubmit = 0x1281,

        /// <summary>
        /// A 64-bit value that describes the current device time counter in nanoseconds when the command identified by the event starts execution on the device.
        /// </summary>
        CommandStart = 0x1282,

        /// <summary>
        /// A 64-bit value that describes the current device time counter in nanoseconds when the command identified by the event has finished execution on the device.
        /// </summary>
        CommandEnd = 0x1283,

        /// <summary>
        /// A 64-bit value that describes the current device time counter in nanoseconds when the command identified by the event and any child commands enqueued by this command on the device have finished execution.
        /// </summary>
        CommandComplete = 0x1284
    }
}