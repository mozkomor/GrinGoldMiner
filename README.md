# GPU Miner for Cuckoo Cycle PoW
This miner currently runs on windows and linux and exist only as a proof of concept for bounty claim. It does not actually interface with grin software yet (see below). It only works on nvidia GPUs (for now, see below) with at least 8GB of RAM. 

 * Stratum server made it to grin and we are working on grin<->miner communication
 * OpenCL version is partly finished with extra speedups, algorithm needs modifications for AMD GCN architecture, more testing and tuning
 * Multiple NV/AMD GPU support with simple config file will be added after that

## Current Perfomance

    GTX 1070    - up to 2.5 Graphs/s @ ~75W       - overclocked core +150 mem +400
    GTX 1070 Ti - up to 2.8 Graphs/s @ ~80W       - overclocked core +150 mem +400
    GTX 1080 Ti - up to 4.4 Graphs/s @ ~200W      - overclocked core +200 mem +400 
    GTX 1080 Ti - up to 3.3 Graphs/s @ ~100W      - power limit 50%
    
Note 1080 Ti uses another .cu file optimized for GPUs with over 8GB VRAM.

## Questions

**Q:** Can I use this to mine grin?

**A:** Not at this moment. Stratum connection to grin is being developed.

**Q:** Can it use multiple GPUs?

**A:** Yes, but this is not enabled yet.

**Q:** Does it run on AMD Vega?

**A:** OpenCL port is being developed.

**Q:** What GPU will be the best for power efficiency or performace per dollar?

**A:** Best to wait for final optimizations and OpenCL version.

## Support

Special thanks to John Tromp, Kristy-Leigh Minehan (OhGodACompany) and Manli Technology for sending us new shiny HW (1080 Ti) for development and tweaking, much appreciated!

License for the software was changed after publication to be compatible with the bounty. Closed sourced miners based on this work will now be required to pay full 50% of the fee to coin developers and keep the other 50%.

If you want to support or speedup development of this cross platform miner, contact us at mozkomorkaban@gmail.com. Happy mining!

----------------------------------

50% bounty here - btc (Photon): 3MRRCuFmS3GciugRRawkuAbLLcpNVXSJqm

50% bounty here - btc (Urza): 3AcRFwqKx6P8rBngUz4hRTG2QEwodLqAVE

Contact us at mozkomorkaban@gmail.com

Based on great work of John Tromp, the inventor of Cuckoo cycle PoW  https://github.com/tromp/cuckoo

And designed for https://github.com/mimblewimble/grin
