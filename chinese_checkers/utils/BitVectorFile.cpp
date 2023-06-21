//
//  BitVectorFile.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/30/13.
//
//

#include "BitVectorFile.h"


BitVectorFile::BitVectorFile(const char *file)
{
	f = fopen(file, "r");
	if (f == 0)
	{
		printf("Error opening '%s'.", file);
	}
	fread(&true_size, sizeof(true_size), 1, f);
	size = (true_size>>storageBitsPower)+1;
	baseOffset = currOffset = sizeof(true_size);
}


BitVectorFile::~BitVectorFile()
{
	fclose(f);
}

bool BitVectorFile::Get(uint64_t index) const
{
	if ((index>>storageBitsPower) > size)
	{
		printf("GET %llu OUT OF RANGE\n", index);
		exit(0);
	}
	uint64_t seekOffset = (index>>storageBitsPower)+baseOffset-currOffset;
	if (fseek(f, seekOffset, SEEK_CUR) != 0)
	{
		printf("Error seeking in file\n");
		exit(0);
	}
	currOffset += seekOffset+1;
	uint8_t value;
	fread(&value, sizeof(uint8_t), 1, f);
	return (((value)>>(index&storageMask))&0x1);
	//return (((storage[index>>storageBitsPower])>>(index&storageMask))&0x1);
}

//void BitVectorFile::Set(uint64_t index, bool value)
//{
//	if ((index>>storageBitsPower) > size)
//	{
//		printf("SET %llu OUT OF RANGE\n", index);
//		exit(0);
//	}
//	if (value)
//		storage[index>>storageBitsPower] = storage[index>>storageBitsPower]|(1<<(index&storageMask));
//	else
//		storage[index>>storageBitsPower] = storage[index>>storageBitsPower]&(~(1<<(index&storageMask)));
//}

