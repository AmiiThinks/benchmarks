//
//  TwoBitFileVector.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/27/14.
//
//

#ifndef __Solve_Chinese_Checkers__TwoBitFileVector__
#define __Solve_Chinese_Checkers__TwoBitFileVector__

#include <stdio.h>
#include <stdint.h>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include "NBitArray.h"
#include <cassert>
#include <cstring>

#define VALIDATE_CALLS 0

// entriesPerGroup is how many 2-bit values are stored in each group
// numGroups is the total number of groups. Size of all data is epg*ng
// entriesPerCache is the number of entries we store in each small cache
// groups per file is how many groups are combined together into each file on disk (to reduce the total number of files)
template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches=2>
class TwoBitFileVector {
public:
	TwoBitFileVector(const char *dataPathPrefix, bool clear);
	~TwoBitFileVector();
	bool NeedsBuild() { return dataFilesEmpty; }
	
	uint64_t Size() const { return (uint64_t)numGroups*(uint64_t)entriesPerGroup; }
	uint8_t Get(uint64_t index) const;
	uint8_t Get(uint64_t groupLoc, uint64_t offset) const;
	bool GetIfInMemory(uint64_t groupLoc, uint64_t offset, uint8_t &result) const;
	void Set(uint64_t groupLoc, uint64_t offset, uint8_t value);
	bool SetIf(uint64_t groupLoc, uint64_t offset, uint8_t newValue, uint8_t oldValue);
	bool SetIfLargeBuffer(uint64_t groupLoc, uint64_t offset, uint8_t newValue, uint8_t oldValue);
	bool SetIfInLargeBuffer(uint64_t groupLoc, uint64_t offset, uint8_t newValue, uint8_t oldValue, bool &inLargeBuffer);
	void LoadReadOnlyBuffer(uint64_t groupLoc);
	void CopyReadOnlyBuffer(uint8_t *readBuffer);
	uint8_t GetReadOnlyBuffer(uint64_t groupLoc, uint64_t offset) const;
	uint8_t GetReadOnlyBuffer(uint64_t groupLoc, uint64_t offset, uint8_t *readBuffer) const;
	bool LoadLargeBufferForWrite(uint64_t whichFile);
	bool IsFileInMemory(uint64_t whichFile) { return fileLargeCacheLRUPtr[whichFile]!=0; }
	bool IsInCache(uint64_t group, uint64_t offset);
private:
	// private, because we don't publicly need to know when we are getting from the large
	// buffer of a disk file or from something else
	uint8_t GetLargeBuffer(uint64_t groupLoc, uint64_t offset) const;
	void SetLargeBuffer(uint64_t groupLoc, uint64_t offset, uint8_t newValue);
	void FlushLargeBufferWrites(uint64_t whichFile);

	void FlushCache(uint64_t group);
	int SelectCacheLine(uint64_t group, uint64_t offset) const;
	int LoadCache(uint64_t group, uint64_t offset) const; // returns cache line
	bool InCache(uint64_t group, uint64_t offset, int &cache) const; // returns cache line
	void GetFileByteOffset(uint64_t group, uint64_t offsetInEntries,
						   int cacheLine,
						   uint64_t &whichFile,
						   uint64_t &fileOffsetOfCurrentCacheInBytes,
						   uint64_t &fileOffsetOfCacheInBytes,
						   uint64_t &offsetOfCacheInCacheEntries,
						   uint64_t &offsetInCacheInBytes,
						   int &offsetInEntryInBits) const;
	void SetupLargeCachePage(uint64_t whichFile);
	
	constexpr static uint64_t GetNumFiles()
	{ return (0 == numGroups%groupsPerFile)?(numGroups/groupsPerFile):(numGroups/groupsPerFile+1);}
	const char *GetFileName(uint64_t whichFile) const;
	
	FILE *files[GetNumFiles()]; // pointers to files for all data

	
	static constexpr uint64_t kEntriesPerByte = 4;
	static constexpr uint64_t kEntriesPerGroupRoundedUp =
	(0==entriesPerGroup%entriesPerCache)?entriesPerGroup:(entriesPerCache-(entriesPerGroup%entriesPerCache)+entriesPerGroup);
	// one full memory cache of a single file
	//uint64_t largeCacheFileID; - trust by not storing this explicitly(!)

	static constexpr uint32_t kNoUser = 0xFFFFFFFF;
	struct cacheLRU {
		cacheLRU *next;
		cacheLRU *prev;
		uint32_t index; // which cache is this representing [out of numLargeCaches]
		uint32_t user; // which file is being cached [back pointer to largeCachePtr]
	};
	//uint8_t largeCache[kEntriesPerGroupRoundedUp/kEntriesPerByte]; // 1.45 MB in final version (1,524,224 bytes)
	// 744 MB in final version (780,346,112 bytes) - assuming 512 groups per file
	// MEMORY: Complete cache of [N] files
	uint8_t largeCaches[numLargeCaches][groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte];
	cacheLRU *fileLargeCacheLRUPtr[GetNumFiles()];
	cacheLRU largeCacheLRUStore[numLargeCaches];
	cacheLRU *largeCachePagesFront, *largeCachePagesBack;
	// LRU changes currently not locked - performed sequentially between groups

	// for writes to large caches during the processing of each group
	//std::mutex largeCacheWriteLock[numLargeCaches][groupsPerFile*kMutexPerGroup];
	
//	int largeCacheID;
	static constexpr int kUseLargeCache = -2;
	static constexpr int kNoCacheID = 0x7FFF;
	static constexpr int kNoLargeCacheID = -1;
	// MEMORY: Read only data for a single group
	uint8_t readOnlyBuffer[kEntriesPerGroupRoundedUp/kEntriesPerByte];
public:
	constexpr static uint64_t GetReadBufferSize() { return kEntriesPerGroupRoundedUp/kEntriesPerByte; }
private:
	struct cacheData {
		unsigned int dirty : 1;
		unsigned int cacheID : 15;
		uint16_t use;
	};
	
	// MEMORY: small caches for each group
	// the caches for each file
	// 6.67 GB in final version (7,163,032,576 bytes) assuming 4x1024 entries. 4x4096 entires = 26 GB
	// 6x(4096ent/4)*6995149 = 6*1024*6995149 = 42,978,195,456
	mutable uint8_t smallCaches[numGroups][cachePerGroup][(entriesPerCache/kEntriesPerByte)]; // 2 bits each
	// the id of what offset is in each cache
	mutable cacheData cacheIDs[numGroups][cachePerGroup]; // 53 MB in final version (55,961,192 bytes)
	// TODO: Add Dirty bit to indicate if we need to write back to disk
	
	
	std::string baseFile;
	bool dataFilesEmpty;
};

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::TwoBitFileVector(const char *file, bool clear)
:baseFile(file)
{
	static_assert(0 == kEntriesPerGroupRoundedUp%entriesPerCache, "Entries per group not rounded up correctly");
	static_assert((groupsPerFile*entriesPerGroup&0x1F) == 0, "groups per file * entries per group should be evenly divisible by 32");

	// data will be stored in 64 bit chunks. Count number of entries and divided by 32 (2 bits per entry)
	static_assert(((entriesPerCache/* *numGroups*/)&0x1F) == 0,
				  "[small memory cache] Error: num caches * entries per cache * num groups should be evenly divisble by 32");

//	largeCacheID = kNoLargeCacheID;
	// 18424 -> 9303 groups
	//(46!/3!43!)((49!/46!3!)/2+91)
	bool fileExists = true;
	FILE *f = fopen(GetFileName(0), "r"); // test open to see if the file is there
	if (f == 0)
	{ fileExists = false; }
	else
	{ fclose(f); }

	if (!fileExists || clear)
	{
		// 1. Create files and clear
		dataFilesEmpty = true;
		for (int x = 0; x < GetNumFiles(); x++)
		{
			const char *file = GetFileName(x);
			int fd = open(file, O_WRONLY|O_CREAT, S_IRWXU);
			if (fd == -1)
			{
				printf("Error %d opening file '%s' [%s]; aborting\n", fd, file, strerror(errno));
				assert(false);
			}
			ftruncate(fd, 0);
			ftruncate(fd, kEntriesPerGroupRoundedUp*groupsPerFile/kEntriesPerByte); // how many bytes for all data; filled with 0
			close(fd);
		}
	}
	else {
		dataFilesEmpty = false;
	}

	// 2. Initialize Cache
	//largeCacheFileID = -1;
	for (int x = 0; x < numGroups; x++)
	{
		for (int y = 0; y < cachePerGroup; y++)
		{
			cacheIDs[x][y].cacheID = kNoCacheID;
			cacheIDs[x][y].dirty = false;
			cacheIDs[x][y].use = 0;
		}
	}
	
	// 3. Open files
	for (int x = 0; x < GetNumFiles(); x++)
	{
		if (clear)
			files[x] = fopen(GetFileName(x), "r+");
		else
			files[x] = fopen(GetFileName(x), "r");
		if (files[x] == 0)
		{
			printf("Error opening '%s' [%s]. Exiting\n", GetFileName(x), strerror(errno));
			assert(false);
		}
	}

	// 4. Set up large cache pages
	for (int x = 0; x < numLargeCaches; x++)
	{
		largeCacheLRUStore[x].user = kNoUser;
		largeCacheLRUStore[x].index = x;
		if (x == 0)
			largeCacheLRUStore[x].prev = 0;
		else
			largeCacheLRUStore[x].prev = &largeCacheLRUStore[x-1];

		if (x == numLargeCaches-1)
			largeCacheLRUStore[x].next = 0;
		else
			largeCacheLRUStore[x].next = &largeCacheLRUStore[x+1];
	}
	largeCachePagesFront = &largeCacheLRUStore[0];
	largeCachePagesBack = &largeCacheLRUStore[numLargeCaches-1];

	for (int x = 0; x < GetNumFiles(); x++)
		fileLargeCacheLRUPtr[x] = 0;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::~TwoBitFileVector()
{
	// TODO: flush all cache
	for (int x = 0; x < GetNumFiles(); x++)
	{
		fclose(files[x]);
		files[x] = 0;
	}

}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
const char *TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::GetFileName(uint64_t whichFile) const
{
	static std::string tmp;
	tmp = baseFile+"CC_EM_"+std::to_string(NUM_SPOTS)+"-"+std::to_string(NUM_PIECES)+"_"+std::to_string(whichFile)+".dat";
	return tmp.c_str();
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
bool TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::IsInCache(uint64_t group, uint64_t offset)
{
	uint64_t whichFile = group/groupsPerFile;
	if (fileLargeCacheLRUPtr[whichFile])
		return true;
	uint64_t neededEntry = offset/entriesPerCache;
	for (int x = 0; x < cachePerGroup; x++)
	{
		if (cacheIDs[group][x].cacheID == neededEntry)
		{
			return true;
		}
	}
	return false;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
int TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::SelectCacheLine(uint64_t group, uint64_t offset) const
{
	uint64_t whichFile = group/groupsPerFile;
	if (fileLargeCacheLRUPtr[whichFile])
		return kUseLargeCache;
//	if (group/groupsPerFile == largeCacheID) //
//		return kUseLargeCache;
	uint64_t neededEntry = offset/entriesPerCache;
	int best = -1;
	uint16_t minVal = 0xFFFF;
	uint16_t maxVal = 0;
	for (int x = 0; x < cachePerGroup; x++)
	{
		minVal = std::min(minVal, cacheIDs[group][x].use);
		maxVal = std::max(maxVal, cacheIDs[group][x].use);
	}
	for (int x = 0; x < cachePerGroup; x++)
	{
		if (cacheIDs[group][x].cacheID == neededEntry)
		{
			cacheIDs[group][x].use = maxVal+1;
			return x;
		}
		if (cacheIDs[group][x].cacheID == kNoCacheID) // unused
			best = x;
	}
	if (best != -1) // found unused
	{
		cacheIDs[group][best].use = maxVal+1;
		return best;
	}
	
	best = 0;
	for (int x = 0; x < cachePerGroup; x++)
	{
		cacheIDs[group][x].use -= minVal;
		if (cacheIDs[group][x].use < cacheIDs[group][best].use)
			best = x;
	}
	return best;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups,
entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::GetFileByteOffset(uint64_t group,
																  uint64_t offsetInEntries,
																  int cacheLine,
																  // TODO: put in which cache here
																  uint64_t &whichFile,
																  uint64_t &fileOffsetOfCurrentCacheInBytes,
																  uint64_t &fileOffsetOfCacheInBytes,
																  uint64_t &offsetOfCacheInCacheEntries,
																  uint64_t &offsetInCacheInBytes,
																  int &offsetInEntryInBits) const
{
	// numFiles = numGroups/groupsPerFile
	whichFile = group/groupsPerFile;
	uint64_t offsetInFileInGroups = group%groupsPerFile;
	uint64_t offsetInFileInEntries = offsetInFileInGroups*kEntriesPerGroupRoundedUp;
	uint64_t offsetInFileInBytes = offsetInFileInEntries/kEntriesPerByte; // Offset of beginning of group
//	uint64_t bytesPerCache = entriesPerCache/kEntriesPerByte;

	fileOffsetOfCurrentCacheInBytes = offsetInFileInBytes+(cacheIDs[group][cacheLine].cacheID*entriesPerCache/kEntriesPerByte);
	offsetOfCacheInCacheEntries = offsetInEntries/entriesPerCache;
	fileOffsetOfCacheInBytes = offsetInFileInBytes+(offsetOfCacheInCacheEntries*entriesPerCache/kEntriesPerByte);
	offsetInCacheInBytes = (offsetInEntries%entriesPerCache)/kEntriesPerByte;
	//(offsetInEntries/kEntriesPerByte)%bytesPerCache;
	offsetInEntryInBits = 2*(offsetInEntries%kEntriesPerByte);
}


//template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
//uint64_t TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::GetCacheOffset(uint64_t offset)
//{
//	return offset/entriesPerCache;
//}

//template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
//void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::FlushAndLoadCache(uint64_t group, uint64_t offset)
//{
//}
template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::FlushCache(uint64_t group)
{
	uint64_t whichFile = group/groupsPerFile;
	assert(fileLargeCacheLRUPtr[whichFile] == 0);
//	if (fileLargeCacheLRUPtr[whichFile]) // //if (whichFile == largeCacheID)
//		return;
	uint64_t offsetInFileInGroups = group%groupsPerFile;
	uint64_t offsetInFileInEntries = offsetInFileInGroups*kEntriesPerGroupRoundedUp;
	uint64_t offsetInFileInBytes = offsetInFileInEntries/kEntriesPerByte; // Offset of beginning of group
//	printf("--Flushing file %d group offset %llu\n", whichFile, offsetInFileInGroups);
	for (int x = 0; x < cachePerGroup; x++)
	{
		if (cacheIDs[group][x].dirty)
		{
			uint64_t fileOffsetOfCurrentCacheInBytes = offsetInFileInBytes+(cacheIDs[group][x].cacheID*entriesPerCache/kEntriesPerByte);

			FILE *f = files[whichFile];//fopen(GetFileName(whichFile), "r+");
			//printf("Flushing to disk offset %llu in file; %llu in cache\n", fileOffsetOfCurrentCacheInBytes, group*entriesPerCache/kEntriesPerByte);
			fseek(f, fileOffsetOfCurrentCacheInBytes, SEEK_SET);
			fwrite(&smallCaches[group][x], 1/*byte*/, entriesPerCache/kEntriesPerByte, f);
		}
		cacheIDs[group][x].cacheID = kNoCacheID;
		cacheIDs[group][x].dirty = false;
		cacheIDs[group][x].use = 0;
	}
}


template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
int TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::LoadCache(uint64_t group, uint64_t offset) const
{
	int whichCache = SelectCacheLine(group, offset);
	if (whichCache == kUseLargeCache)
		return kUseLargeCache;
	uint64_t whichFile, offsetInCacheInBytes;
	uint64_t fileOffsetOfCacheInBytes, fileOffsetOfCurrentCacheInBytes, offsetOfCacheInCacheEntries;
	int offsetInEntryInBits;
	GetFileByteOffset(group, offset,
					  whichCache,
					  whichFile,
					  fileOffsetOfCurrentCacheInBytes,
					  fileOffsetOfCacheInBytes,
					  offsetOfCacheInCacheEntries,
					  offsetInCacheInBytes,
					  offsetInEntryInBits);

//	static int hits = 0, misses = 0;
//	const int whichGroup = 10;

	if (cacheIDs[group][whichCache].cacheID == offsetOfCacheInCacheEntries)
	{
//		if (group == whichGroup)
//			hits++;
//		cacheIDs[group][whichCache].use = currTime++;
		return whichCache;
	}
	
//	if (group == whichGroup)
//	{
//		misses++;
//		//printf("[%llu] Group %d hits: %d; misses %d.\n", offsetOfCacheInCacheEntries, whichGroup, hits, misses);
//	}
	FILE *f = files[whichFile];//fopen(GetFileName(whichFile), "r+");
	if (f == 0)
	{
		printf("Error opening '%s' [%s]. Exiting\n", GetFileName(whichFile), strerror(errno));
		assert(false);
	}
	if (cacheIDs[group][whichCache].dirty)
	{
		//printf("Flushing to disk offset %llu in file; %llu in cache\n", fileOffsetOfCurrentCacheInBytes, group*entriesPerCache/kEntriesPerByte);
		fseek(f, fileOffsetOfCurrentCacheInBytes, SEEK_SET);
		fwrite(&smallCaches[group][whichCache], 1/*byte*/, entriesPerCache/kEntriesPerByte, f);
	}
	
	cacheIDs[group][whichCache].cacheID = static_cast<unsigned int>(offsetOfCacheInCacheEntries);
	cacheIDs[group][whichCache].dirty = false;
	//cacheIDs[group][whichCache].use = currTime++;
	fseek(f, fileOffsetOfCacheInBytes, SEEK_SET);
	fread(&smallCaches[group][whichCache], 1/*byte*/, entriesPerCache/kEntriesPerByte, f);
	//printf("Reading from disk offset %llu in file; %llu in cache\n", fileOffsetOfCacheInBytes, group*entriesPerCache/kEntriesPerByte);
	//fclose(f);
	return whichCache;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
bool TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::InCache(uint64_t group, uint64_t offset, int &whichCache) const
{
	whichCache = SelectCacheLine(group, offset);
	if (whichCache == kUseLargeCache)
		return true;
	uint64_t whichFile, offsetInCacheInBytes;
	uint64_t fileOffsetOfCacheInBytes, fileOffsetOfCurrentCacheInBytes, offsetOfCacheInCacheEntries;
	int offsetInEntryInBits;
	GetFileByteOffset(group, offset,
					  whichCache,
					  whichFile,
					  fileOffsetOfCurrentCacheInBytes,
					  fileOffsetOfCacheInBytes,
					  offsetOfCacheInCacheEntries,
					  offsetInCacheInBytes,
					  offsetInEntryInBits);
	
	if (cacheIDs[group][whichCache].cacheID == offsetOfCacheInCacheEntries)
	{
		return true;
	}
	return false;
}


/* Loads single group into memory
 * All subsequent calls to GetLargeBuffer use this memory, and do not error check
 * Data is never cleared nor written back to disk
 */
template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::LoadReadOnlyBuffer(uint64_t groupLoc)
{
	uint64_t whichFile = groupLoc/groupsPerFile;
	uint64_t offsetInFileInGroups = groupLoc%groupsPerFile;
	uint64_t offsetInFileInEntries = offsetInFileInGroups*kEntriesPerGroupRoundedUp;
	uint64_t offsetInFileInBytes = offsetInFileInEntries/kEntriesPerByte; // Offset of beginning of group

//	if (whichFile == largeCacheID)
	if (fileLargeCacheLRUPtr[whichFile])
	{
		// get from memory via memcpy
		//assert(false);
		memcpy(readOnlyBuffer, &largeCaches[fileLargeCacheLRUPtr[whichFile]->index][offsetInFileInBytes], kEntriesPerGroupRoundedUp/kEntriesPerByte);
		//memcpy(readOnlyBuffer, &largeCache[offsetInFileInBytes], kEntriesPerGroupRoundedUp/kEntriesPerByte);
		return;
	}
	
	// just loads the particular group we are working on
	FlushCache(groupLoc);


	FILE *f = files[whichFile];//fopen(GetFileName(whichFile), "r+");
	if (f == 0)
	{
		printf("Error opening '%s' [%s]. Exiting\n", GetFileName(whichFile), strerror(errno));
		assert(false);
	}
	fseek(f, offsetInFileInBytes, SEEK_SET);
	fread(readOnlyBuffer, 1/*byte*/, kEntriesPerGroupRoundedUp/kEntriesPerByte, f);
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::CopyReadOnlyBuffer(uint8_t *readBuffer)
{
	memcpy(readBuffer, readOnlyBuffer, kEntriesPerGroupRoundedUp/kEntriesPerByte);
}


/* Setup memory page for large cache
 */
template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::SetupLargeCachePage(uint64_t whichFile)
{
	assert(fileLargeCacheLRUPtr[whichFile] == 0);
	
	// 0. Clean up page at back of LRU
	if (largeCachePagesBack->user != kNoUser)
	{
		//printf("[Kick %d] ", largeCachePagesBack->user);
		// Writes to disk, removes pointer to LRU
		FlushLargeBufferWrites(largeCachePagesBack->user);
		largeCachePagesBack->user = kNoUser;
	}
	
	// 1. Move page from back of LRU to front
	fileLargeCacheLRUPtr[whichFile] = largeCachePagesBack;
	largeCachePagesBack = largeCachePagesBack->prev;
	largeCachePagesBack->next = 0;

	fileLargeCacheLRUPtr[whichFile]->next = largeCachePagesFront;
	largeCachePagesFront->prev = fileLargeCacheLRUPtr[whichFile];
	fileLargeCacheLRUPtr[whichFile]->prev = 0;
	largeCachePagesFront = fileLargeCacheLRUPtr[whichFile];
	
	// 2. Take ownership for our use
	fileLargeCacheLRUPtr[whichFile]->user = whichFile;
}

/* Loads a full file into memory
 * We assume that no other calls are made between loading/unloading the large buffer besides SetIfLargeBuffer
 * These functions are provided to allow efficient writing of data from the disk cache.
 */
template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
bool TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::LoadLargeBufferForWrite(uint64_t whichFile)
{
	if (fileLargeCacheLRUPtr[whichFile])
	{
		if (fileLargeCacheLRUPtr[whichFile]->prev == 0) // already at front
			return false;
		
		// Extract from queue
		fileLargeCacheLRUPtr[whichFile]->prev->next = fileLargeCacheLRUPtr[whichFile]->next;
		if (fileLargeCacheLRUPtr[whichFile]->next != 0) // not at end
			fileLargeCacheLRUPtr[whichFile]->next->prev = fileLargeCacheLRUPtr[whichFile]->prev;
		else // at end of LRU queue
			largeCachePagesBack = fileLargeCacheLRUPtr[whichFile]->prev;
		
		// move to front
		fileLargeCacheLRUPtr[whichFile]->next = largeCachePagesFront;
		fileLargeCacheLRUPtr[whichFile]->next->prev = fileLargeCacheLRUPtr[whichFile];
		fileLargeCacheLRUPtr[whichFile]->prev = 0;
		largeCachePagesFront = fileLargeCacheLRUPtr[whichFile];
		return false;
	}
//	if (largeCacheID == whichFile)
//		return;

	//FlushLargeBufferWrites(largeCacheID);
	
	//FlushLargeBufferWrites(whichFile);
//	printf("Loading file %d for sustained writes\n", whichFile);

	// TODO: flushing old caches - but it would be more efficient to load to memory and then
	// flush to memory, instead of flushing to disk and then loading from disk
	for (uint64_t x = whichFile*groupsPerFile; x < numGroups && x < (whichFile+1)*groupsPerFile; x++)
		FlushCache(x);
	
	SetupLargeCachePage(whichFile);

	FILE *f = files[whichFile];//fopen(GetFileName(whichFile), "r+");
	if (f == 0)
	{
		printf("Error opening '%s' [%s]. Exiting\n", GetFileName(whichFile), strerror(errno));
		assert(false);
	}
	fseek(f, 0, SEEK_SET);
	fread(largeCaches[fileLargeCacheLRUPtr[whichFile]->index], 1/*byte*/, groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte, f);
	return true; // loaded
	//	largeCacheID = whichFile;
}

/* Writes a full file back to disk
 * We assume that no other calls are made between loading/unloading the large buffer besides SetIfLargeBuffer
 * These functions are provided to allow efficient writing of data from the disk cache.
 * [Note: the assumption about other calls may no longer be true.]
 * [We aren't are now flushing when we load the next chunk.]
 */
template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::FlushLargeBufferWrites(uint64_t whichFile)
{
	// Nothing for this file cached - we only flush if there is a cache
	assert(fileLargeCacheLRUPtr[whichFile] != 0);

//	if (whichFile == kNoLargeCacheID)
//		return;
//	if (VALIDATE_CALLS)
//		assert(whichFile == largeCacheID);
	FILE *f = files[whichFile];//fopen(GetFileName(whichFile), "r+");
	if (f == 0)
	{
		printf("Error opening '%s' [%s]. Exiting\n", GetFileName(whichFile), strerror(errno));
		assert(false);
	}
	fseek(f, 0, SEEK_SET);
	fwrite(largeCaches[fileLargeCacheLRUPtr[whichFile]->index], 1/*byte*/, groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte, f);
	fileLargeCacheLRUPtr[whichFile] = 0; // we are done

	//fwrite(largeCache, 1/*byte*/, groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte, f);
	
	//largeCacheID = kNoLargeCacheID;
	//	printf("        file %d ending sustained writes\n", whichFile);
}



template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
uint8_t TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::Get(uint64_t index) const
{
	return Get(index/entriesPerGroup, index%entriesPerGroup);
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
uint8_t TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::Get(uint64_t groupLoc, uint64_t offset) const
{
	int whichCache = LoadCache(groupLoc, offset);
	if (whichCache == kUseLargeCache)
		return GetLargeBuffer(groupLoc, offset);
	uint64_t whichFile, offsetInCacheInBytes;
	uint64_t fileOffsetOfCacheInBytes, fileOffsetOfCurrentCacheInBytes, offsetOfCacheInCacheEntries;
	int offsetInEntryInBits;
	GetFileByteOffset(groupLoc, offset, whichCache,
					  whichFile,
					  fileOffsetOfCurrentCacheInBytes,
					  fileOffsetOfCacheInBytes,
					  offsetOfCacheInCacheEntries,
					  offsetInCacheInBytes,
					  offsetInEntryInBits);

	uint8_t result = (smallCaches[groupLoc][whichCache][offsetInCacheInBytes]>>offsetInEntryInBits)&0x3;
	//printf("[%llu] Read %llu / %llu: %d\n", groupLoc*entriesPerCache/kEntriesPerByte+offsetInCacheInBytes,groupLoc, offset, result);
	return result;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
bool TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::GetIfInMemory(uint64_t groupLoc, uint64_t offset, uint8_t &result) const
{
	int whichCache;
	if (!InCache(groupLoc, offset, whichCache))
		return false;
	if (whichCache == kUseLargeCache)
	{
		result = GetLargeBuffer(groupLoc, offset);
		return true;
	}
	uint64_t whichFile, offsetInCacheInBytes;
	uint64_t fileOffsetOfCacheInBytes, fileOffsetOfCurrentCacheInBytes, offsetOfCacheInCacheEntries;
	int offsetInEntryInBits;
	GetFileByteOffset(groupLoc, offset, whichCache,
					  whichFile,
					  fileOffsetOfCurrentCacheInBytes,
					  fileOffsetOfCacheInBytes,
					  offsetOfCacheInCacheEntries,
					  offsetInCacheInBytes,
					  offsetInEntryInBits);
	
	result = (smallCaches[groupLoc][whichCache][offsetInCacheInBytes]>>offsetInEntryInBits)&0x3;
	//printf("[%llu] Read %llu / %llu: %d\n", groupLoc*entriesPerCache/kEntriesPerByte+offsetInCacheInBytes,groupLoc, offset, result);
	return true;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
uint8_t TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::GetLargeBuffer(uint64_t groupLoc, uint64_t offset) const
{
	uint64_t whichFile = groupLoc/groupsPerFile;
	uint64_t groupOffsetInBuffer = groupLoc%groupsPerFile;//groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte
	uint64_t byteOffsetOfGroupInBuffer = groupOffsetInBuffer*kEntriesPerGroupRoundedUp/kEntriesPerByte;
	uint64_t offsetInCacheInBytes = offset/kEntriesPerByte;
	uint64_t offsetInEntryInBits = 2*(offset%kEntriesPerByte);
//	uint8_t result = ((largeCache[byteOffsetOfGroupInBuffer+offsetInCacheInBytes])>>offsetInEntryInBits)&0x3;
	assert(fileLargeCacheLRUPtr[whichFile] != 0);
	uint8_t result = ((largeCaches[fileLargeCacheLRUPtr[whichFile]->index][byteOffsetOfGroupInBuffer+offsetInCacheInBytes])>>offsetInEntryInBits)&0x3;
	return result;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
uint8_t TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::GetReadOnlyBuffer(uint64_t groupLoc, uint64_t offset) const
{
	uint8_t result = (readOnlyBuffer[offset/kEntriesPerByte]>>(2*(offset%kEntriesPerByte)))&0x3;
	return result;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
uint8_t TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::GetReadOnlyBuffer(uint64_t groupLoc, uint64_t offset, uint8_t *readBuffer) const
{
	uint8_t result = (readBuffer[offset/kEntriesPerByte]>>(2*(offset%kEntriesPerByte)))&0x3;
	return result;
}


template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::Set(uint64_t groupLoc, uint64_t offset, uint8_t value)
{
	int whichCache = LoadCache(groupLoc, offset);
	if (whichCache == kUseLargeCache)
		return SetLargeBuffer(groupLoc, offset, value);

	uint64_t whichFile, offsetInCacheInBytes;
	uint64_t fileOffsetOfCacheInBytes, fileOffsetOfCurrentCacheInBytes, offsetOfCacheInCacheEntries;
	int offsetInEntryInBits;
	GetFileByteOffset(groupLoc, offset, whichCache,
					  whichFile,
					  fileOffsetOfCurrentCacheInBytes,
					  fileOffsetOfCacheInBytes,
					  offsetOfCacheInCacheEntries,
					  offsetInCacheInBytes,
					  offsetInEntryInBits);

	uint8_t result = (smallCaches[groupLoc][whichCache][offsetInCacheInBytes]);
	result &= ~(0x3<<offsetInEntryInBits);
	result |= (value<<offsetInEntryInBits);
	smallCaches[groupLoc][whichCache][offsetInCacheInBytes] = result;//<<offsetInEntryInBits)&0x3;
	cacheIDs[groupLoc][whichCache].dirty = true;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
bool TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::SetIf(uint64_t groupLoc, uint64_t offset,
																										uint8_t newValue, uint8_t oldValue)
{
	int whichCache = LoadCache(groupLoc, offset);
	if (whichCache == kUseLargeCache)
		return SetIfLargeBuffer(groupLoc, offset, newValue, oldValue);

	uint64_t whichFile, offsetInCacheInBytes;
	uint64_t fileOffsetOfCacheInBytes, fileOffsetOfCurrentCacheInBytes, offsetOfCacheInCacheEntries;
	int offsetInEntryInBits;
	GetFileByteOffset(groupLoc, offset, whichCache,
					  whichFile,
					  fileOffsetOfCurrentCacheInBytes,
					  fileOffsetOfCacheInBytes,
					  offsetOfCacheInCacheEntries,
					  offsetInCacheInBytes,
					  offsetInEntryInBits);

	uint8_t result = (smallCaches[groupLoc][whichCache][offsetInCacheInBytes]);
	if (((result>>offsetInEntryInBits)&0x3) == oldValue)
	{
		result &= ~(0x3<<offsetInEntryInBits);
		result |= (newValue<<offsetInEntryInBits);
		smallCaches[groupLoc][whichCache][offsetInCacheInBytes] = result;//<<offsetInEntryInBits)&0x3;
		cacheIDs[groupLoc][whichCache].dirty = true;
		//printf("WriteIF %llu / %llu: %d\n", groupLoc, offset, newValue);
		return true;
	}
	return false;
	//	Get(groupLoc, offset);

}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
bool TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::SetIfLargeBuffer(uint64_t groupLoc, uint64_t offset, uint8_t newValue, uint8_t oldValue)
{
	uint64_t whichFile = groupLoc/groupsPerFile;
	uint64_t groupOffsetInBuffer = groupLoc%groupsPerFile;//groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte
	uint64_t byteOffsetOfGroupInBuffer = groupOffsetInBuffer*kEntriesPerGroupRoundedUp/kEntriesPerByte;
	uint64_t offsetInCacheInBytes = offset/kEntriesPerByte;
	uint64_t offsetInEntryInBits = 2*(offset%kEntriesPerByte);
	assert(fileLargeCacheLRUPtr[whichFile] != 0);
	
//	std::lock_guard<std::mutex> guard(largeCacheWriteLock[fileLargeCacheLRUPtr[whichFile]->index][groupOffsetInBuffer*kMutexPerGroup+offset/kEntriesPerMutex]);
	uint8_t result = largeCaches[fileLargeCacheLRUPtr[whichFile]->index][byteOffsetOfGroupInBuffer+offsetInCacheInBytes];
	if (((result>>offsetInEntryInBits)&0x3) == oldValue)
	{
		result &= ~(0x3<<offsetInEntryInBits);
		result |= (newValue<<offsetInEntryInBits);
		largeCaches[fileLargeCacheLRUPtr[whichFile]->index][byteOffsetOfGroupInBuffer+offsetInCacheInBytes] = result;
		return true;
	}
	return false;
}

template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
bool TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::SetIfInLargeBuffer(uint64_t groupLoc, uint64_t offset, uint8_t newValue, uint8_t oldValue, bool &inBuffer)
{
	uint64_t whichFile = groupLoc/groupsPerFile;
	if (fileLargeCacheLRUPtr[whichFile] == 0) //if (groupLoc/groupsPerFile != largeCacheID)
	{
		inBuffer = false;
		return false;
	}
	// TODO: Need locking on write
	inBuffer = true;
	uint64_t groupOffsetInBuffer = groupLoc%groupsPerFile;//groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte
	uint64_t byteOffsetOfGroupInBuffer = groupOffsetInBuffer*kEntriesPerGroupRoundedUp/kEntriesPerByte;
	uint64_t offsetInCacheInBytes = offset/kEntriesPerByte;
	uint64_t offsetInEntryInBits = 2*(offset%kEntriesPerByte);

//	std::lock_guard<std::mutex> guard(largeCacheWriteLock[fileLargeCacheLRUPtr[whichFile]->index][groupOffsetInBuffer*kMutexPerGroup+offset/kEntriesPerMutex]);
	
	uint8_t result = largeCaches[fileLargeCacheLRUPtr[whichFile]->index][byteOffsetOfGroupInBuffer+offsetInCacheInBytes];
	if (((result>>offsetInEntryInBits)&0x3) == oldValue)
	{
		result &= ~(0x3<<offsetInEntryInBits);
		result |= (newValue<<offsetInEntryInBits);
		largeCaches[fileLargeCacheLRUPtr[whichFile]->index][byteOffsetOfGroupInBuffer+offsetInCacheInBytes] = result;
		return true;
	}
	return false;
}


template <uint32_t entriesPerGroup, int numGroups, int entriesPerCache, int cachePerGroup, int groupsPerFile, int numLargeCaches>
void TwoBitFileVector<entriesPerGroup, numGroups, entriesPerCache, cachePerGroup, groupsPerFile, numLargeCaches>::SetLargeBuffer(uint64_t groupLoc,
																																 uint64_t offset,
																																 uint8_t newValue)
{
	uint64_t whichFile = groupLoc/groupsPerFile;
	uint64_t groupOffsetInBuffer = groupLoc%groupsPerFile;//groupsPerFile*kEntriesPerGroupRoundedUp/kEntriesPerByte
	uint64_t byteOffsetOfGroupInBuffer = groupOffsetInBuffer*kEntriesPerGroupRoundedUp/kEntriesPerByte;
	uint64_t offsetInCacheInBytes = offset/kEntriesPerByte;
	uint64_t offsetInEntryInBits = 2*(offset%kEntriesPerByte);
	assert(fileLargeCacheLRUPtr[whichFile] != 0);
	
//	std::lock_guard<std::mutex> guard(largeCacheWriteLock[fileLargeCacheLRUPtr[whichFile]->index][groupOffsetInBuffer*kMutexPerGroup+offset/kEntriesPerMutex]);
	uint8_t result = largeCaches[fileLargeCacheLRUPtr[whichFile]->index][byteOffsetOfGroupInBuffer+offsetInCacheInBytes];
	result &= ~(0x3<<offsetInEntryInBits);
	result |= (newValue<<offsetInEntryInBits);
	largeCaches[fileLargeCacheLRUPtr[whichFile]->index][byteOffsetOfGroupInBuffer+offsetInCacheInBytes] = result;
}


#endif /* defined(__Solve_Chinese_Checkers__TwoBitFileVector__) */


