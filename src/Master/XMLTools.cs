// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml.Serialization;

namespace Mozkomor.GrinGoldMiner
{
    public static class XmlTools
    {
        public static string ToXmlString<T>(this T input)
        {
            using (var writer = new StringWriter())
            {
                input.ToXml(writer);
                return writer.ToString();
            }
        }
        public static void ToXml<T>(this T objectToSerialize, Stream stream)
        {
            new XmlSerializer(typeof(T)).Serialize(stream, objectToSerialize);
        }

        public static void ToXml<T>(this T objectToSerialize, StringWriter writer)
        {
            new XmlSerializer(typeof(T)).Serialize(writer, objectToSerialize);
        }
    }

    public class Serialization
    {
        public static bool Serialize<T>(T input, string outputFile)
        {
            try
            {
                // Serialization
                XmlSerializer s = new XmlSerializer(typeof(T));
                using (TextWriter w = new StreamWriter(outputFile))
                {
                    s.Serialize(w, input);
                }

                return true;
            }
            catch (Exception Ex)
            {
                Logger.Log(Ex);
                return false;
            }
        }

        public static T DeSerialize<T>(string inputFile)
        {
            //try
            //{
                // Deserialization
                XmlSerializer s = new XmlSerializer(typeof(T));
                T newClass;
                using (TextReader r = new StreamReader(inputFile))
                {
                    newClass = (T)s.Deserialize(r);
                }

                return newClass;
            //}
            //catch (Exception Ex)
            //{
            //    Logger.Log(Ex);
            //    return default(T);
            //}
        }

        public static T DeSerializeString<T>(string inputContent)
        {
            //try
            //{
                // Deserialization
                XmlSerializer s = new XmlSerializer(typeof(T));
                T newClass;
                using (TextReader r = new StringReader(inputContent))
                {
                    newClass = (T)s.Deserialize(r);
                }

                return newClass;
            //}
            //catch (Exception Ex)
            //{
            //    Logger.Log(Ex);
            //    return default(T);
            //}
        }
    }
}
