
namespace OpenCl.DotNetCore.CommandQueues
{
    /// <summary>
    /// Represents an enumeration for the status of the execution of a command.
    /// </summary>
    public enum CommandExecutionStatus : int
    {
        /// <summary>
        /// The command could not be executed successfully.
        /// </summary>
        Error = -0x1,

        /// <summary>
        /// The command has completed.
        /// </summary>
        Complete = 0x0,

        /// <summary>
        /// The device is currently executing this command.
        /// </summary>
        Running = 0x1,

        /// <summary>
        /// The enqueued command has been submitted by the host to the device associated with the command-queue.
        /// </summary>
        Submitted = 0x2,

        /// <summary>
        /// The command has been enqueued in the command-queue.
        /// </summary>
        Queued = 0x3
    }
}