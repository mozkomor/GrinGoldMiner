// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Kukacka optimized miner is covered by the FAIR MINING license


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GGM
{
    public unsafe struct DValue
    {
        public fixed uint vals[2];
        public fixed ushort keys[2];
        public byte cnt;
    }

    public class Cuckonary
    {
        const int dedges = 2;
        const int dBits = 17;
        const int sBits = 12;
        const uint sMask = (1 << sBits) - 1;
        const uint eMask = (1 << 29) - 1;

        private Dictionary<uint, uint> dic = new Dictionary<uint, uint>();
        private DValue[] data;
        private int v;

        public Cuckonary()
        {
            data = new DValue[1 << dBits];
        }

        public Cuckonary(int v)
        {
            this.v = v;

            data = new DValue[1 << dBits];
        }

        public unsafe void Add(uint key, uint value)
        {
            unchecked
            {
                uint idx = (key & (eMask)) >> sBits;
                ushort k = (ushort)((key & (eMask)) & sMask);
                var hit = data[idx];

                if (hit.cnt < dedges)
                {
                    for (int i = 0; i < dedges; i++)
                    {
                        if (hit.keys[i] == 0)
                        {
                            hit.keys[i] = k;
                            hit.vals[i] = value;

                            break;
                        }
                    }

                    hit.cnt++;
                }
                else
                {
                    dic[key] = value;
                }

                data[idx] = hit;
            }
        }

        public unsafe bool ContainsKey(uint key)
        {
            unchecked
            {
                uint idx = (key & (eMask)) >> sBits;
                ushort k = (ushort)((key & (eMask)) & sMask);
                var hit = data[idx];

                if ((hit.keys[0] == k || hit.keys[1] == k))
                    return true;

                if (hit.cnt == dedges)
                    return dic.ContainsKey(key);
                else
                    return false;
            }
        }

        public unsafe void Remove(uint key)
        {
            unchecked
            {
                uint idx = (key & (eMask)) >> sBits;
                ushort k = (ushort)((key & (eMask)) & sMask);
                var hit = data[idx];

                bool removed = false;
                for (int i = 0; i < dedges; i++)
                {
                    if (hit.keys[i] == k)
                    {
                        hit.keys[i] = 0;
                        hit.vals[i] = 0;
                        removed = true;
                    }
                }

                if (!removed)
                    dic.Remove(key);
            }
        }

        public unsafe uint this[uint key]
        {
            get
            {
                unchecked
                {
                    uint idx = (key & (eMask)) >> sBits;
                    ushort k = (ushort)((key & (eMask)) & sMask);
                    var hit = data[idx];

                    for (int i = 0; i < dedges; i++)
                    {
                        if (hit.keys[i] == k)
                        {
                            return hit.vals[i];
                        }
                    }

                    if (hit.cnt == dedges)
                        return dic[key];

                    return 0;
                }
            }
            set
            {
                Add(key, value);
            }
        }
    }
}
