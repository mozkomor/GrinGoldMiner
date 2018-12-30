using System;
using System.Collections.Generic;
using System.Text;

namespace OclSolver
{
    public class Comms
    {
    }

    public struct Job
    {
        public DateTime timestamp;
        public UInt64 nonce, height, difficulty, jobID;
        public UInt64 k0;
        public UInt64 k1;
        public UInt64 k2;
        public UInt64 k3;
    }

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

    public class Solution
    {
        public Job job;
        public List<Edge> nonces;
    }
}
