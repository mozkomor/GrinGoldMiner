using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SharedSerialization;

namespace CudaSolver
{
    public class Comms
    {
        public static Queue<Solution> graphSolutionsOut = new Queue<Solution>();
        public static Queue<LogMessage> logsOut = new Queue<LogMessage>();
        public static GpuDevicesMessage gpuMsg = null;

        public static AutoResetEvent flushToMaster;
        private static TcpClient client;
        private static NetworkStream stream;
        private static Task listener;
        private static Task sender;
        private static int _port;
        public static Job nextJob;
        public static volatile bool IsTerminated = false;

        internal static void ConnectToMaster(int port)
        {
            _port = port;
            flushToMaster = new AutoResetEvent(false);
            client = new TcpClient("127.0.0.1", port);
            if (client.Connected)
            {
                stream = client.GetStream();
                listener = Task.Factory.StartNew(() => { Listen(); }, TaskCreationOptions.LongRunning);
                sender = Task.Factory.StartNew(() => { WaitSend(); }, TaskCreationOptions.LongRunning);
            }
        }

        private static void WaitSend()
        {
            while (!IsTerminated)
            {
                try
                {
                    flushToMaster.WaitOne();

                    if (graphSolutionsOut.Count > 0)
                    {
                        Solution s;
                        lock (graphSolutionsOut)
                        {
                            s = graphSolutionsOut.Dequeue();
                        }
                        (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, s);
                    }
                    if (gpuMsg != null)
                    {
                        (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, gpuMsg);
                        gpuMsg = null;
                    }
                    if (logsOut.Count > 0)
                    {
                        LogMessage lm;
                        lock (logsOut)
                        {
                            lm = logsOut.Dequeue();
                        }
                         (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, lm);
                    }
                }
                catch (Exception ex)
                {
                    // log to local file and console
                    //Logger.Log(LogLevel.Warning, "WaitSend error", ex);
                }
            }
        }

        private static void Listen()
        {
            while (!IsTerminated)
            {
                try
                {
                    object payload = (new BinaryFormatter()).Deserialize(stream);

                    switch (payload)
                    {
                        case Job job:
                            nextJob = job;
                            Console.WriteLine($"New job received: {job.pre_pow}");
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log(LogLevel.Warning, "Listen error", ex);
                }
            }
        }

        internal static bool IsConnected()
        {
            return client != null && client.Connected;
        }

        internal static void SetEvent()
        {
            if (flushToMaster != null)
                flushToMaster.Set();
        }

        internal static void Close()
        {
            try
            {
                IsTerminated = true;
                if (stream != null)
                    stream.Close();
                if (client != null)
                    client.Close();
            }
            catch { }
        }
    }




}
