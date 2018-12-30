
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumeration for the different affinity domains devices may have.
    /// </summary>
    [Flags]
    public enum DeviceAffinityDomain : ulong
    {
        /// <summary>
        /// Split the device into sub-devices comprised of compute units that share a NUMA node.
        /// </summary>
        Numa = 1 << 0,

        /// <summary>
        /// Split the device into sub-devices comprised of compute units that share a level 4 data cache.
        /// </summary>
        Level4Cache = 1 << 1,
        
        /// <summary>
        /// Split the device into sub-devices comprised of compute units that share a level 3 data cache.
        /// </summary>
        Level3Cache = 1 << 2,
        
        /// <summary>
        /// Split the device into sub-devices comprised of compute units that share a level 2 data cache.
        /// </summary>
        Level2Cache = 1 << 3,
        
        /// <summary>
        /// Split the device into sub-devices comprised of compute units that share a level 1 data cache.
        /// </summary>
        Level1Cache = 1 << 4,
        
        /// <summary>
        /// Split the device along the next partitionable affinity domain. The implementation shall find the first level along which the device or sub-device may be further subdivided in the order NUMA, L4, L3, L2, L1, and partition the
        /// device into sub-devices comprised of compute units that share memory subsystems at this level.
        /// </summary>
        NextPartitionable = 1 << 5
    }
}