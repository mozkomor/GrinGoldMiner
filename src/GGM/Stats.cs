// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license


using System;
using System.Collections.Generic;
using System.Text;

namespace GGM
{
    public class Stats
    {
        public int graphs = 0;
        public int edgesets = 0;
        public int solutions = 0;
        public int mined = 0;
    }
}
