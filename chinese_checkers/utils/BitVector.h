// HOG File
/*
 * $Id: BitVector.h,v 1.4 2006/09/18 06:20:15 nathanst Exp $
 *
 * This file is part of HOG.
 *
 * HOG is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * HOG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with HOG; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/ 

#ifndef _BITVECTOR_
#define _BITVECTOR_

#include <stdint.h>
#include "MMapUtil.h"
#include <sys/mman.h>

/**
 * An efficient bit-wise vector implementation.
 */


// Note: this doesn't work. Bug somewhere.
//typedef uint64_t storageElement;
//const int storageBits = 64;
//const int storageBitsPower = 6;
//const int storageMask = 0x3F;

//typedef uint32_t storageElement;
//const int storageBits = 32;
//const int storageBitsPower = 5;
//const int storageMask = 0x1F;

// don't change this -- it makes the file reading/writing much easier
// and it will cause the BitVectorFile to fail
typedef uint8_t storageElement;
const int storageBits = 8;
const int storageBitsPower = 3;
const int storageMask = 0x7;


class BitVector {
public:
	BitVector(uint64_t size);
	BitVector(const char *file);
	BitVector(uint64_t entries, const char *file, bool zero);
	~BitVector();
//	BitVector& operator=(const BitVector&); // TODO: implement the rule of 3(5)
	void Advise(int val);
	void clear();
//	BitVector *Clone();
	void Resize(uint64_t size);
	uint64_t GetSize() { return true_size; }
	bool Get(uint64_t index) const;
	void Set(uint64_t index, bool value);
//	void Merge(BitVector *);
	bool Save(const char *);
	bool Load(const char *);
	bool Equals(BitVector *);
	int GetNumSetBits();
private:
	uint64_t size, true_size;
	storageElement *storage;
	bool memmap;
	int fd;
	char fname[255];
};

#endif
