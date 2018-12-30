
namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from an pipe object.
    /// </summary>
    public enum PipeInformation : uint
    {
        /// <summary>
        /// Retrieves the pipe packet size specified when pipe is created with <see cref="CreatePipe"/>.
        /// </summary>
        PacketSize = 0x1120,

        /// <summary>
        /// Retrieves the maximum number of packets specified when pipe is created with <see cref="CreatePipe"/>.
        /// </summary>
        MaximumNumberOfPackets = 0x1121
    }
}