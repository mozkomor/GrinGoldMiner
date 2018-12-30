
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Events
{
    /// <summary>
    /// Represents an OpenCL event, which has been created by the user and is not bound to a command enqueued on the command queue.
    /// </summary>
    public class UserEvent : AwaitableEvent
    {
        #region Constructors

        /// <summary>
        /// Initializes a new <see cref="UserEvent"/> instance.
        /// </summary>
        /// <param name="handle">The handle to the OpenCL event.</param>
        public UserEvent(IntPtr handle)
            : base(handle)
        {
        }

        #endregion
    }
}