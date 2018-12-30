
namespace OpenCl.DotNetCore.Interop.Events
{
    /// <summary>
    /// Represents an enumeration that identifies the event information that can be queried from an event.
    /// </summary>
    public enum EventInformation : uint
    {
        /// <summary>
        /// The command-queue associated with the event. For user event objects, a <c>null</c> value is returned. If the cl_khr_gl_sharing extension is enabled, the command queue of a linked event <c>null</c>, because the event is not associated
        /// with any OpenCL command queue. If the cl_khr_egl_event extension is enabled, the CL_EVENT_COMMAND_QUEUE of a linked event is <c>null</c>, because the event is not associated with any OpenCL command queue.
        /// </summary>
        CommandQueue = 0x11D0,

        /// <summary>
        /// The command associated with the event.
        /// </summary>
        CommandType = 0x11D1,

        /// <summary>
        /// The event reference count. The reference count returned should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.
        /// </summary>
        ReferenceCount = 0x11D2,

        /// <summary>
        /// The execution status of the command identified by the event.
        /// </summary>
        CommandExecutionStatus = 0x11D3,

        /// <summary>
        /// The context associated with the event.
        /// </summary>
        Context = 0x11D4
    }
}