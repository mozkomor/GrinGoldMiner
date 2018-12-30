
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumeration for the different floating point configurations of devices.
    /// </summary>
    [Flags]
    public enum DeviceFloatingPointConfiguration : ulong
    {
        /// <summary>
        /// Denorms are supported.
        /// </summary>
        Denorm = 1 << 0,

        /// <summary>
        /// Inifinity (INF) and Not-a-Number's (NaNs) are supported.
        /// </summary>
        InfinityAndNotANumber = 1 << 1,
        
        /// <summary>
        /// Round to nearest even rounding mode is supported.
        /// </summary>
        RoundToNearest = 1 << 2,
        
        /// <summary>
        /// Round to zero rounding mode supported.
        /// </summary>
        RoundToZero = 1 << 3,
        
        /// <summary>
        /// Round to +ve and -ve infinity rounding modes supported.
        /// </summary>
        RoundToInfinity = 1 << 4,
        
        /// <summary>
        /// IEEE754-2008 fused multiply-add is supported.
        /// </summary>
        FusedMultiplyAdd = 1 << 5,
        
        /// <summary>
        /// Basic floating-point operations (such as addition, subtraction, multiplication) are implemented in software.
        /// </summary>
        SoftwareFloat = 1 << 6,
        
        /// <summary>
        /// Divide and sqrt are correctly rounded as defined by the IEEE754 specification.
        /// </summary>
        CorrectlyRoundedDivideSquareRoot = 1 << 7
    }
}