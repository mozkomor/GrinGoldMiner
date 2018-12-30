
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Memory
{
    /// <summary>
    /// Represents an OpenCL memory buffer.
    /// </summary>
    public class MemoryBuffer : MemoryObject
    {
        #region Constructors

        /// <summary>
        /// Initializes a new <see cref="Buffer"/> instance.
        /// </summary>
        /// <param name="handle">The handle to the OpenCL memory buffer.</param>
        public MemoryBuffer(IntPtr handle)
            : base(handle)
        {
        }

        #endregion
    }
}