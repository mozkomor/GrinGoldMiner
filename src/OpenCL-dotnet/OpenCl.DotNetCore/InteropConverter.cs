
#region Using Directives

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

#endregion

namespace OpenCl.DotNetCore
{
    /// <summary>
    /// Represents a converter, which is used to convert byte arrays into several different CLR data types.
    /// </summary>
    public static class InteropConverter
    {
        #region Private Static Fields

        /// <summary>
        /// Contains a map, which maps a data type to a converter that is able to convert it.
        /// </summary>
        private static Dictionary<Type, Func<byte[], object>> converterMap = new Dictionary<Type, Func<byte[], object>>();
        
        #endregion

        #region Public Static Methods

        /// <summary>
        /// Converts the specified data to the data type specified as type parameter.
        /// </summary>
        /// <typeparam name="T">The data type into which the data is to be converted.</typeparam>
        /// <param name="data">The data that is to be converted.</param>
        /// <exception cref="InvalidOperationException">If no fitting convert could be found, then an <see cref="InvalidOperationException"/> is thrown.</exception>
        /// <returns>Returns the converted data.</returns>
        public static T To<T>(byte[] data)
        {
            if (typeof(T) == typeof(string))
                return (T)(object)ToString(data);

            // Checks if there is a converter for the specified data type
            Type targetType = typeof(T);
            if (!InteropConverter.converterMap.ContainsKey(targetType))
            {
                // Since the converter could not be found, a fitting method is searched, if one is found, then it is added to the converter map, otherwise an exception is thrown
                TypeInfo typeInfo = typeof(InteropConverter).GetTypeInfo();
                MethodInfo converterMethodInfo = typeInfo.GetMethods().FirstOrDefault(method =>
                    method.IsStatic &&
                    method.ReturnType == targetType &&
                    method.GetParameters().Count() == 1 &&
                    method.GetParameters().First().ParameterType == typeof(byte[]) &&
                    method.Name == $"To{targetType.Name}");
                if (converterMethodInfo == null)
                    throw new InvalidOperationException($"No fitting converter could be found for the type {targetType.Name}");
                InteropConverter.converterMap.Add(targetType, dataToConvert => converterMethodInfo.Invoke(null, new object[] { dataToConvert }));
            }

            // Gets the fitting converter, converts the value and returns it
            Func<byte[], object> converter = InteropConverter.converterMap[targetType];
            return (T)converter(data);
        }


        /// <summary>
        /// Converts a byte array to a 32 bit integer value.
        /// </summary>
        /// <param name="data">The byte array that is to be converted.</param>
        /// <returns>Returns the converted data.</returns>
        public static int ToInt32(byte[] data) => BitConverter.ToInt32(data, 0);

        /// <summary>
        /// Converts a byte array to a 32 bit unsigned integer value.
        /// </summary>
        /// <param name="data">The byte array that is to be converted.</param>
        /// <returns>Returns the converted data.</returns>
        public static uint ToUInt32(byte[] data) => BitConverter.ToUInt32(data, 0);

        /// <summary>
        /// Converts a byte array to a 64 bit integer value.
        /// </summary>
        /// <param name="data">The byte array that is to be converted.</param>
        /// <returns>Returns the converted data.</returns>
        public static long ToInt64(byte[] data) => BitConverter.ToInt64(data, 0);

        /// <summary>
        /// Converts a byte array to a 64 bit unsigned integer value.
        /// </summary>
        /// <param name="data">The byte array that is to be converted.</param>
        /// <returns>Returns the converted data.</returns>
        public static ulong ToUInt64(byte[] data) => BitConverter.ToUInt64(data, 0);

        /// <summary>
        /// Converts a byte array to a string value.
        /// </summary>
        /// <param name="data">The byte array that is to be converted.</param>
        /// <returns>Returns the converted data.</returns>
        public static string ToString(byte[] data)
        {
            // All OpenCL String are ASCII encoded, therefore the byte array is decoded using the ASCII encoder
            string result = Encoding.ASCII.GetString(data);

            // All OpenCL strings are null-terminated, therefore the null-terminator must be removed
            result = result.Replace("\0", string.Empty);

            // Returns the converted string
            return result;
        }

        #endregion
    }
}