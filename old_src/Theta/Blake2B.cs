/*	BLAKE2.cs source code package - C# implementation

	Written in 2012 by Samuel Neves <sneves@dei.uc.pt>
	Written in 2012 by Christian Winnerlein <codesinchaos@gmail.com>
	Written in 2016 by Uli Riehm <metadings@live.de>

	To the extent possible under law, the author(s) have dedicated all copyright
	and related and neighboring rights to this software to the public domain
	worldwide. This software is distributed without any warranty.

	You should have received a copy of the CC0 Public Domain Dedication along with
	this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
*/
using System;
using System.Security.Cryptography;

namespace Crypto
{
	public partial class Blake2B : HashAlgorithm //, IDisposable
	{
		public static ulong BytesToUInt64(byte[] buf, int offset)
		{
			return
				((ulong)buf[offset + 7] << 7 * 8 |
				((ulong)buf[offset + 6] << 6 * 8) |
				((ulong)buf[offset + 5] << 5 * 8) |
				((ulong)buf[offset + 4] << 4 * 8) |
				((ulong)buf[offset + 3] << 3 * 8) |
				((ulong)buf[offset + 2] << 2 * 8) |
				((ulong)buf[offset + 1] << 1 * 8) |
				((ulong)buf[offset]));
		}

		public static void UInt64ToBytes(ulong value, byte[] buf, int offset)
		{
			buf[offset + 7] = (byte)(value >> 7 * 8);
			buf[offset + 6] = (byte)(value >> 6 * 8);
			buf[offset + 5] = (byte)(value >> 5 * 8);
			buf[offset + 4] = (byte)(value >> 4 * 8);
			buf[offset + 3] = (byte)(value >> 3 * 8);
			buf[offset + 2] = (byte)(value >> 2 * 8);
			buf[offset + 1] = (byte)(value >> 1 * 8);
			buf[offset] = (byte)value;
		}

		private readonly int hashSize = 512;

		public override int HashSize { get { return hashSize; } }

		public int HashSizeInBytes { get { return hashSize / 8; } }

		public int HashSizeInUInt64 { get { return HashSizeInBytes / 4; } }

		public Blake2B()
		{
			fanOut = 1;
			maxHeight = 1;
			// leafSize = 0;
			// intermediateHashSize = 0;
		}

		public Blake2B(int hashSizeInBits)
			: this()
		{	
			if (hashSizeInBits < 8) // || hashSizeInBits > 512)
				throw new ArgumentOutOfRangeException("hashSizeInBits");
			if (hashSizeInBits % 8 != 0)
				throw new ArgumentOutOfRangeException("hashSizeInBits", "MUST be a multiple of 8");
			
			hashSize = hashSizeInBits;
		}

		// enum blake2b_constant's
		public const int BLAKE2B_BLOCKBYTES = 128;
		public const int BLAKE2B_BLOCKUINT64S = BLAKE2B_BLOCKBYTES / 8;
		public const int BLAKE2B_OUTBYTES = 64;
		public const int BLAKE2B_KEYBYTES = 64;
		public const int BLAKE2B_SALTBYTES = 16;
		public const int BLAKE2B_PERSONALBYTES = 16;

		public const ulong IV0 = 0x6A09E667F3BCC908UL;
		public const ulong IV1 = 0xBB67AE8584CAA73BUL;
		public const ulong IV2 = 0x3C6EF372FE94F82BUL;
		public const ulong IV3 = 0xA54FF53A5F1D36F1UL;
		public const ulong IV4 = 0x510E527FADE682D1UL;
		public const ulong IV5 = 0x9B05688C2B3E6C1FUL;
		public const ulong IV6 = 0x1F83D9ABFB41BD6BUL;
		public const ulong IV7 = 0x5BE0CD19137E2179UL;

		private bool isInitialized = false;

		private int bufferFilled;
		private byte[] buffer = new byte[BLAKE2B_BLOCKBYTES];
		private ulong[] state = new ulong[8];
		private ulong[] m = new ulong[16];
		private ulong counter0;
		private ulong counter1;
		private ulong f0;
		private ulong f1;

		public const int ROUNDS = 12;

		public static readonly int[] Sigma = new int[ROUNDS * 16] {
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
			11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4,
			7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8,
			9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13,
			2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9,
			12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11,
			13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
			6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
			10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3
		};

		public virtual ulong[] Prepare()
		{
			var c = new ulong[8];

			// digest length
			c[0] |= (uint)HashSizeInBytes;

			// Key length
			if (Key != null)
			{
				if (Key.Length > BLAKE2B_KEYBYTES)
					throw new ArgumentException("Key", "Key too long");

				c[0] |= ((ulong)Key.Length << 8);
			}

			if (IntermediateHashSize > 64)
				throw new ArgumentOutOfRangeException("IntermediateHashSize");

			// bool isSequential = TreeConfig == null;
			// FanOut
			c[0] |= FanOut << 16;
			// Depth
			c[0] |= MaxHeight << 24;
			// Leaf length
			c[0] |= LeafSize << 32;
			// Inner length
			c[2] |= IntermediateHashSize << 8;

			// Salt
			if (Salt != null)
			{
				if (Salt.Length != BLAKE2B_SALTBYTES)
					throw new ArgumentException("Salt has invalid length");

				c[4] = BytesToUInt64(Salt, 0);
				c[5] = BytesToUInt64(Salt, 8);
			}
			// Personalization
			if (Personalization != null)
			{
				if (Personalization.Length != BLAKE2B_PERSONALBYTES)
					throw new ArgumentException("Personalization has invalid length");

				c[6] = BytesToUInt64(Personalization, 0);
				c[7] = BytesToUInt64(Personalization, 8);
			}

			return c;
		}

		private ulong[] rawConfig;

		public override void Initialize()
		{
			if (rawConfig == null)
			{
				rawConfig = Prepare();
			}
			Initialize(rawConfig);
		}

		/* public static void ConfigBSetNode(ulong[] rawConfig, byte depth, ulong nodeOffset)
		{
			rawConfig[1] = nodeOffset;
			rawConfig[2] = (rawConfig[2] & ~0xFFul) | depth;
		} */

		public virtual void Initialize(ulong[] c)
		{
			if (c == null)
				throw new ArgumentNullException("config");
			if (c.Length != 8)
				throw new ArgumentException("config length must be 8 words");

			HashClear();

			state[0] = IV0;
			state[1] = IV1;
			state[2] = IV2;
			state[3] = IV3;
			state[4] = IV4;
			state[5] = IV5;
			state[6] = IV6;
			state[7] = IV7;

			for (int i = 0; i < 8; i++) state[i] ^= c[i];

			isInitialized = true;

			if (Key != null) HashCore(Key, 0, Key.Length);
		}

		// public void Dispose() { Dispose(true); }

		protected override void Dispose(bool disposing) { if (disposing) HashClear(); base.Dispose(disposing); }

		public virtual void HashClear()
		{
			isInitialized = false;

			counter0 = 0UL;
			counter1 = 0UL;
			f0 = 0UL;
			f1 = 0UL;

			bufferFilled = 0;
			int i;
			for (i = 0; i < BLAKE2B_BLOCKBYTES; ++i) buffer[i] = 0x00;
			for (i = 0; i < 8; ++i) state[i] = 0UL;
			for (i = 0; i < 16; ++i) m[i] = 0UL;
		}

		protected bool IsLastNode { get { return f1 == ulong.MaxValue; } }

		protected void SetLastNode() { f1 = ulong.MaxValue; }

		protected void ClearLastNode() { f1 = 0; }

		protected bool IsLastBlock { get { return f0 == ulong.MaxValue; } }

		protected void SetLastBlock()
		{
			if( IsLastNode ) SetLastNode();
			f0 = ulong.MaxValue;
		}

		protected void ClearLastBlock()
		{
			if( IsLastNode ) ClearLastNode();
			f0 = 0;
		}

		protected void IncrementCounter( ulong inc )
		{
			counter0 += inc;
			if (counter0 == 0) ++counter1;
		}

		protected override void HashCore(byte[] array, int offset, int length)
		{
			Core(array, offset, length);
		}

		public virtual void Core(byte[] array, int offset, int length)
		{
			if (array == null)
				throw new ArgumentNullException("array");
			if (offset < 0)
				throw new ArgumentOutOfRangeException("offset");
			if (length < 0)
				throw new ArgumentOutOfRangeException("length");
			if (offset + length > array.Length)
				throw new ArgumentOutOfRangeException("offset + length");

			if (!isInitialized) Initialize();

			int bytesToFill;
			while (0 < length)
			{
				bytesToFill = Math.Min(length, BLAKE2B_BLOCKBYTES - bufferFilled);
				Buffer.BlockCopy(array, offset, buffer, bufferFilled, bytesToFill);

				bufferFilled += bytesToFill;
				offset += bytesToFill;
				length -= bytesToFill;

				if (bufferFilled == BLAKE2B_BLOCKBYTES)
				{
					IncrementCounter((ulong)BLAKE2B_BLOCKBYTES);

					if (BitConverter.IsLittleEndian)
						Buffer.BlockCopy(buffer, 0, m, 0, BLAKE2B_BLOCKBYTES);
					else
						for (int i = 0; i < BLAKE2B_BLOCKUINT64S; ++i)
							m[i] = BytesToUInt64(buffer, (i << 3));

					Compress();

					bufferFilled = 0;
				}
			}
		}

		partial void Compress();

		protected override byte[] HashFinal()
		{
			return Final();
		}

		public virtual byte[] Final()
		{
			var hash = new byte[HashSizeInBytes];
			Final(hash);
			return hash;
		}

		/* public virtual byte[] Final(bool isEndOfLayer)
		{
			var hash = new byte[HashSizeInBytes];
			Final(hash, isEndOfLayer);
			return hash;
		}

		public virtual void Final(byte[] hash)
		{
			Final(hash, false);
		} /**/

		public virtual void Final(byte[] hash) //, bool isEndOfLayer)
		{
			if (hash.Length != HashSizeInBytes)
				throw new ArgumentOutOfRangeException("hash", 
					string.Format("hash.Length must be {0} HashSizeInBytes",
						HashSizeInBytes));

			if (!isInitialized) Initialize();

			// Last compression
			IncrementCounter((ulong)bufferFilled);

			SetLastBlock();

			for (int i = bufferFilled; i < BLAKE2B_BLOCKBYTES; ++i) buffer[i] = 0x00;

			if (BitConverter.IsLittleEndian)
				Buffer.BlockCopy(buffer, 0, m, 0, BLAKE2B_BLOCKBYTES);
			else
				for (int i = 0; i < BLAKE2B_BLOCKUINT64S; ++i)
					m[i] = BytesToUInt64(buffer, (i << 3));
			
			Compress();

			// Output
			if (BitConverter.IsLittleEndian)
				Buffer.BlockCopy(state, 0, hash, 0, HashSizeInBytes);
			else
				for (int i = 0; i < HashSizeInUInt64; ++i)
					UInt64ToBytes(state[i], hash, i << 3);

			isInitialized = false;
		}

		public virtual void Compute(byte[] value, byte[] sourceCode)
		{
			Core(sourceCode, 0, sourceCode.Length);
			Final(value);
		}

		public virtual byte[] Compute(byte[] sourceCode)
		{
			Core(sourceCode, 0, sourceCode.Length);
			return Final();
		}

		public override byte[] Hash
		{
			get {
				// if (m_bDisposed) throw new ObjectDisposedException(null);
				// if (State != 0) throw new CryptographicUnexpectedOperationException(Environment.GetResourceString("Cryptography_HashNotYetFinalized"));

				// Output
				var hash = new byte[HashSizeInBytes];
				if (BitConverter.IsLittleEndian)
					Buffer.BlockCopy(state, 0, hash, 0, HashSizeInBytes);
				else
					for (int i = 0; i < HashSizeInUInt64; ++i)
						UInt64ToBytes(state[i], hash, i << 3);
				return hash;
			}
		}


		private uint fanOut;

		public uint FanOut
		{ 
			get { return fanOut; }
			set { 
				fanOut = value; 
				rawConfig = null;
				isInitialized = false;
			}
		}

		private uint maxHeight;

		public uint MaxHeight
		{ 
			get { return maxHeight; }
			set { 
				maxHeight = value; 
				rawConfig = null;
				isInitialized = false;
			}
		}

		private ulong leafSize;

		public ulong LeafSize
		{ 
			get { return leafSize; }
			set { 
				leafSize = value; 
				rawConfig = null;
				isInitialized = false;
			}
		}

		private uint intermediateHashSize;

		public uint IntermediateHashSize
		{ 
			get { return intermediateHashSize; }
			set { 
				intermediateHashSize = value; 
				rawConfig = null;
				isInitialized = false;
			}
		}


		private byte[] personalization;

		public byte[] Personalization 
		{ 
			get { return personalization; }
			set { 
				personalization = value; 
				rawConfig = null;
				isInitialized = false;
			}
		}

		private byte[] salt;

		public byte[] Salt 
		{ 
			get { return salt; }
			set { 
				salt = value; 
				rawConfig = null;
				isInitialized = false;
			}
		}

		private byte[] key;

		public byte[] Key
		{ 
			get { return key; }
			set { 
				key = value; 
				rawConfig = null;
				isInitialized = false;
			}
		}

	}
}
