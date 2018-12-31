using System;
using System.Collections.Generic;
using System.Linq;

namespace SharedData
{

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

        public ulong[] GetUlongEdges()
        {
            return edges.Select(e => (ulong)e.Item1 | (((ulong)e.Item2) << 32)).ToArray();
        }
    }

    [SerializableAttribute]
    public class GpuDevice
    {
        public int id;
        public string name;
        public long memory;
    }
    [SerializableAttribute]
    public class GpuDevicesMessage
    {
        public List<GpuDevice> devices;
    }
}
