
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop
{
    /// <summary>
    /// Represents a custom attribute, which can be used to mark an element of the public API with the OpenCL version that is was introduced in.
    /// </summary>
    [AttributeUsage(AttributeTargets.Enum | AttributeTargets.Field | AttributeTargets.Method | AttributeTargets.Struct, Inherited = true, AllowMultiple = false)]
    public class IntroducedInOpenClAttribute : Attribute
    {
        #region Constructors

        /// <summary>
        /// Initializes a new <see cref="IntroducedInOpenClAttribute"/> instance.
        /// </summary>
        /// <param name="majorVersion">The major version of OpenCL in which the marked element of the public API was introduced.</param>
        /// <param name="minorVersion">The minor version of OpenCL in which the marked element of the public API was introduced.</param>
        public IntroducedInOpenClAttribute(int majorVersion, int minorVersion)
        {
            this.MajorVersion = majorVersion;
            this.MinorVersion = minorVersion;
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets or sets the major version of OpenCL in which the marked element of the public API was introduced.
        /// </summary>
        public int MajorVersion { get; private set; }

        /// <summary>
        /// Gets or sets the minor version of OpenCL in which the marked element of the public API was introduced.
        /// </summary>
        public int MinorVersion { get; private set; }

        #endregion
    }
}