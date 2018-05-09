# GPU Miner for Cuckoo Cycle PoW - Work In Progress
This miner runs on windows and linux. It does not actually interface with grin software yet (see below). OpenCL version is being made, bounty cuda version can be seen in repo. 

 * Stratum communication tested, implementation underway
 * OpenCL version is finished but needs to be rewritten for AMD to work around a feature we did not know about (credits to OhGodACompany) - currently too slow on AMD
 * Multiple NV/AMD GPU support with simple config file will be added after that

## Expected Perfomance (OpenCL)

    GTX 1070    - up to 2.9 gps
    GTX 1070 Ti - up to 3.2 gps
    GTX 1080 Ti - up to 5.2 Graphs/s @ ~250W
    GTX 1080 Ti - up to 4.0 Graphs/s @ ~125W
    Vega64      - unknown (1080Ti > Vega64 > 1070)
    RX580       - unknown - do not have
    Intel iGPU  - in future update
    
## Questions

**Q:** Can I use this to mine testnet2 grin?

**A:** Not at this moment. Stratum connection to grin is being developed next.

**Q:** Can it use multiple GPUs?

**A:** Yes, but this is not enabled yet. Multiple instances can be executed in the menatime.

**Q:** Does it run on AMD Vega?

**A:** OpenCL port is working, but we need to fix poor AMD performance.

**Q:** What GPU will be the best for power efficiency or performace per dollar?

**A:** For this implementation, next gen nvidia. But we need multiple independent miners from different developers to make conclusions. This is only a guess.

## Support

Special thanks to John Tromp, Kristy-Leigh Minehan (OhGodACompany) and Manli Technology for sending us new shiny HW (1080 Ti) for development and tweaking + helping with slow AMD code, much appreciated!

License for the software was changed after publication to be compatible with the bounty. Closed sourced miners based on this work will now be required to pay full 50% of the fee to coin developers and keep the other 50%.

If you want to support or speedup development of this cross platform miner, contact us at mozkomorkaban@gmail.com. Happy mining!

----------------------------------

4x speedup bounty has been received. No further bounty claims will be made.

50% bounty here - btc (Photon): 3MRRCuFmS3GciugRRawkuAbLLcpNVXSJqm

50% bounty here - btc (Urza): 3AcRFwqKx6P8rBngUz4hRTG2QEwodLqAVE

Contact us at mozkomorkaban@gmail.com

Based on great work of John Tromp, the inventor of Cuckoo cycle PoW  https://github.com/tromp/cuckoo

And designed for https://github.com/mimblewimble/grin
