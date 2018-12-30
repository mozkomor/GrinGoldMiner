
namespace OpenCl.DotNetCore.Interop.Kernels
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from an OpenCL kernel sub group.
    /// </summary>
    public enum KernelSubGroupInformation : uint
    {
        /// <summary>
        /// Returns the maximum sub-group size for the kernel. All subgroups must be the same size, while the last sub-group in any work-group (i.e. the sub-group with the maximum index) could be the same or smaller size. The input value must
        /// be an array of <see cref="IntPtr"/> values corresponding to the local work size parameter of the intended dispatch. The number of dimensions in the ND-range will be inferred from the value specified for input value size.
        /// </summary>
        MaximumSubGroupSizeForNDRange = 0x2033,

        /// <summary>
        /// Returns the number of sub-groups that will be present in each work-group for a given local work size. All workgroups, apart from the last work-group in each dimension in the presence of non-uniform work-group sizes, will have the
        /// same number of subgroups. The input value must be an array of <see cref="IntPtr"/> values corresponding to the local work size parameter of the intended dispatch. The number of dimensions in the ND-range will be inferred from the
        /// value specified for input value size.
        /// </summary>
        SubGroupCountForNDRange = 0x2034,

        /// <summary>
        /// Returns the local size that will generate the requested number of sub-groups for the kernel. The output array must be an array of <see cref="IntPtr"/> values corresponding to the local size parameter. Any returned work-group will
        /// have one dimension. Other dimensions inferred from the value specified for parameter value size will be filled with the value 1. The returned value will produce an exact number of sub-groups and result in no partial groups for an
        /// executing kernel except in the case where the last work-group in a dimension has a size different from that of the other groups. If no work-group size can accommodate the requested number of sub-groups, 0 will be returned in each
        /// element of the return array.
        /// </summary>
        LocalSizeForSubGroupCount = 0x11B8
    }
}