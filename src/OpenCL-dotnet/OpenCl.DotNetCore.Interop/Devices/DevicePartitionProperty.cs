
namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumeration for the different device partition properties.
    /// </summary>
    public enum DevicePartitionProperty : uint
    {
        /// <summary>
        /// Partitions the device equally among sub-devices.
        /// </summary>
        PartitionEqually = 0x1086,

        /// <summary>
        /// Partitions the device among sub-devices by counts.
        /// </summary>
        PartitionByCounts = 0x1087,

        /// <summary>
        /// Marks the end of the <see cref="PartitionByCounts"/> list.
        /// </summary>
        PartitionByCountsListEnd = 0x0,

        /// <summary>
        /// Partitions the device among sub-devices along a cache line.
        /// </summary>
        PartitionByAffinityDomain = 0x1088
    }
}