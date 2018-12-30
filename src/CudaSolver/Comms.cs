using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Google.Protobuf;

namespace CudaSolver
{
    public class Comms
    {
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
            //while (!IsTerminated)   
            (new BinaryFormatter()).Serialize(stream, new Solution());
        }

        private static void Listen()
        {
            object john = (new BinaryFormatter()).Deserialize(stream);
            switch (john)
            {
                case Solution sol when sol.job.height > 5:
                    break;
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
    }


    [SerializableAttribute]
    public struct Job
    {
        public DateTime timestamp;
        public UInt64 nonce, height, difficulty, jobID;
        public UInt64 k0;
	    public UInt64 k1;
	    public UInt64 k2;
	    public UInt64 k3;
    }
    [SerializableAttribute]
    public struct Edge
    {
        public Edge(UInt32 U, UInt32 V)
        {
            Item1 = U;
            Item2 = V;
        }

        public UInt32 Item1;
        public UInt32 Item2;
    }
    [SerializableAttribute]
    public class Solution
    {
        public Job job;
        public List<Edge> edges;
        public UInt32[] nonces;

        internal ulong[] GetUlongEdges()
        {
            return edges.Select(e => (ulong)e.Item1 | (((ulong)e.Item2) << 32)).ToArray();
        }
    }
}
