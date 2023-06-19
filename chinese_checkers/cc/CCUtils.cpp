//
//  CCUtils.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/20/13.
//
//

#include "CCUtils.h"
#include "CCheckers.h"
#include "Timer.h"
#include <cassert>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "Memory.h"

const uint32_t maxFileEntries = 1<<30;

int GetBackPieceAdvancement(const CCState &s, int who)
{
	if (who == 0)
	{
		int result = s.pieces[0][NUM_PIECES-1];
		return result;
	}
	if (who == 1)
	{
		int result = NUM_SPOTS-1-s.pieces[1][0];
		return result;
	}
	assert(false);
	return 0;
}


int GetDepth(const char *prefix, CCState &s, int who)
{
	CCheckers cc;
	CCState flipped;
	int64_t r1, r2;
	//assert(BITS == 4);
	const int BITS = 4;
	if (who == 0)
		cc.rankPlayer(s, 0, r1, r2);
	else
		cc.rankPlayerFlipped(s, 1, r1, r2);
	char fname[255];
	uint8_t value;
	sprintf(fname, "%sCC-%d-%d-b%d-f%lld.bin", prefix, NUM_SPOTS, NUM_PIECES, int(r1), r2/maxFileEntries);
	FILE *f = fopen(fname, "r");
	if (f == 0)
	{
		if (who == 0)
		{
			cc.FlipPlayer(s, flipped, 0);
			cc.rankPlayer(flipped, 0, r1, r2);
		}
		else {
			cc.FlipPlayer(s, flipped, 1);
			cc.rankPlayerFlipped(flipped, 1, r1, r2);
		}
		
		sprintf(fname, "%sCC-%d-%d-b%d-f%lld.bin", prefix, NUM_SPOTS, NUM_PIECES, int(r1), r2/maxFileEntries);
		f = fopen(fname, "r");
		if (f == 0)
		{
			return -1;
			//printf("Error opening file %s\n", fname);
			//exit(0);
		}
	}
	fseek(f, (r2%maxFileEntries)/2, SEEK_SET);
	fread(&value, 1, 1, f);
	fclose(f);
	if ((r2%2) == 0)
		return (value>>0)&0xF;
	else
		return (value>>4)&0xF;
}


void LocalSearch(const std::vector<bucketInfo> &data, std::vector<bucketStructure> &structure,
				 std::vector<int> &readOrder, int count);
void GeneticAlgorithm(const std::vector<bucketInfo> &data, std::vector<bucketStructure> &structure,
					  std::vector<int> &readOrder);

void InitTwoPieceData(std::vector<bucketInfo> &data, uint64_t maxBucketSize, int openSize, bool SYMMETRY)
{
	CCheckers cc;
	CCState s;
	
	int64_t maxVal2 = cc.getMaxSinglePlayerRank2();
	int64_t maxRank = cc.getMaxSinglePlayerRank();
	
	data.resize(maxVal2);
	
	uint64_t bucketSize = 0;
	
	int currBucket = 0;
	
	// build data sets
	for (int64_t x = 0; x < maxRank; x++)
	{
		cc.unrankPlayer(x, s, 0);
		
		int64_t r1;
		cc.rankPlayerFirstTwo(s, 0, r1);
		
		int64_t maxValOther = cc.getMaxSinglePlayerRank2(r1);
		x+=maxValOther-1;
		
		if (SYMMETRY && cc.symmetricStart(s))
		{
			data[r1].unused = true;
			data[r1].numEntries = maxValOther;
			data[r1].bucketID = -1;
		}
		else {
			if (bucketSize > maxBucketSize)
			{
				bucketSize = 0;
				currBucket++;
			}
			data[r1].unused = false;
			data[r1].bucketID = currBucket;
			data[r1].numEntries = maxValOther;
			data[r1].bucketOffset = bucketSize;
			bucketSize += maxValOther;
			// 64-bit align data in buckets
			while (bucketSize%openSize != 0)
				bucketSize++;
		}
	}
}

// go through all
void EvalOrdering(const std::vector<bucketInfo> &data, std::vector<bucketStructure> &structure,
				  const std::vector<int> &readOrder, int64_t &maxMemory, int &maxIndex, bool print = false);

int64_t InitTwoPieceStructure(const std::vector<bucketInfo> &data, int nb,
						   std::vector<bucketStructure> &structure,
						   std::vector<int> &readOrder, bool SYMMETRY, int iterCount)
{
	readOrder.resize(0);
	Timer t;
	t.StartTimer();
	CCheckers cc;
	CCState s;
	
	//static bool once = false;
	structure.resize(data.size());
	for (int x = 0; x < structure.size(); x++)
	{
		structure[x].touchesBucket.resize(nb);
		structure[x].touchedGroups.resize(data.size());
		structure[x].firstGroup = data.size();
		structure[x].lastGroup = 0;
	}
//	printf("1\n");
	uint64_t count = cc.getMaxSinglePlayerRank(4);
	int64_t parentRank2;
	int64_t neighborRank2;
	for (uint64_t rank = 0; rank < count; rank++)
	{
		cc.unrankPlayer(rank, s, 0, 4);
		if (!cc.validInFullBoard(s, 4))
		{
			//			printf("Skipping: ");
			//			s.PrintASCII();
			continue;
		}
		
		//		s.PrintASCII();
		//cc.rankPlayerFirstTwo(s, 0, parentRank2, 4);
		cc.rankPlayerFirstTwoInFullBoard(s, 0, parentRank2, 4);
		
		//printf("2Rank: %llu\n", parentRank2);
		for (int x = 0; x < 4; x++)
		{
			CCMove *m = cc.getMovesForPiece(s, x);
			//			printf("%d: %d moves\n", x, m->length());
			for (CCMove *tt = m; tt; tt = tt->next)
			{
				cc.ApplyMove(s, tt);
				if (cc.validInFullBoard(s, 4))
				{
					//					s.PrintASCII();
					cc.rankPlayerFirstTwoInFullBoard(s, 0, neighborRank2, 4);
					//					printf("Setting [%llu] [%llu] (bucket %d)\n", parentRank2, neighborRank2, data[neighborRank2].bucketID);
					if (data[neighborRank2].bucketID != -1)
					{
						structure[parentRank2].firstGroup = std::min(structure[parentRank2].firstGroup, neighborRank2);
						structure[parentRank2].lastGroup = std::max(structure[parentRank2].lastGroup, neighborRank2);
						structure[parentRank2].touchesBucket[data[neighborRank2].bucketID] = true;
						structure[parentRank2].touchedGroups[neighborRank2] = true;
					}
				}
				if (SYMMETRY)
				{
					CCState d;
					cc.FlipPlayer(s, d, 0);
					if (cc.validInFullBoard(d, 4))
					{
						cc.rankPlayerFirstTwoInFullBoard(d, 0, neighborRank2, 4);
						if (data[neighborRank2].bucketID != -1)
						{
							structure[parentRank2].firstGroup = std::min(structure[parentRank2].firstGroup, neighborRank2);
							structure[parentRank2].lastGroup = std::max(structure[parentRank2].lastGroup, neighborRank2);
							structure[parentRank2].touchesBucket[data[neighborRank2].bucketID] = true;
						}
					}
				}
				cc.UndoMove(s, tt);
			}
			cc.freeMove(m);
		}
	}
//	printf("2\n");

	for (int x = 0; x < structure.size(); x++)
	{
		for (int y = 0; y < structure[x].touchedGroups.size(); y++)
		{
			if (structure[x].touchedGroups[y])
			{
				structure[x].touchedGroupsID.push_back(y);
			}
		}
	}
//	printf("3\n");

	// Compute neighbors for each group
	int64_t largestRange = 0;
	int64_t largestGroup = 0;
	for (int x = 0; x < structure.size(); x++)
	{
		if (data[x].unused)
			continue;
		int64_t sum = 0;
		int64_t sum2 = 0;
		int touched = 0;
		for (int y = structure[x].firstGroup; y <= structure[x].lastGroup; y++)
		{
			if (!data[y].unused)
			{
				sum += data[y].numEntries;
				if (structure[x].touchedGroups[y])
				{
					sum2 += data[y].numEntries;
					touched++;
				}
			}
		}
		structure[x].numNeighbors = touched;
		//		printf("[%7lld] [%7lld] Group %3d [%llu items / %d neighbors] goes from %lld to %lld and hits buckets: ",
		//			   sum, sum2,
		//			   x, data[x].numEntries, touched,
		//			   structure[x].firstGroup, structure[x].lastGroup);
		largestRange = std::max(largestRange, sum);
		largestGroup = std::max(largestGroup, sum2);
		//		for (int y = 0; y < structure[x].touchesBucket.size(); y++)
		//		{
		//			if (structure[x].touchesBucket[y])
		//			{
		//				printf("%d ", y);
		//			}
		//		}
		//		printf("\n");
	}
//	printf("Largest range: %lld entries\n", largestRange);
//	printf("Largest group: %lld entries\n", largestGroup);
	printf("%1.2fs doing pre-processing\nBeginning optimization.\n", t.EndTimer());
	
	t.StartTimer();
	//GeneticAlgorithm(data, structure, readOrder);
	LocalSearch(data, structure, readOrder, iterCount);
	int64_t maxMem;
	int maxIndex;
	printf("%1.2fs running optimization\n", t.EndTimer());
		EvalOrdering(data, structure, readOrder, maxMem, maxIndex, true);
	printf("Reached maximum memory usage of %1.2fGB (%1.2fMB) at step %d\n", maxMem/1024.0/1024.0/1024.0,
		   maxMem/1024.0/1024.0, maxIndex);
	printf("%lu items in read order\n", readOrder.size());
	return maxMem;
}

void GeneticAlgorithm(const std::vector<bucketInfo> &data, std::vector<bucketStructure> &structure,
					  std::vector<int> &readOrder)
{
	const int popSize = 50;
	const int bestCutoff = 8;
	const int iterations = 1000;
	const int mutations = 4;
	std::vector<std::vector<int> > currPopulation(popSize);
	std::vector<std::vector<int> > nextPopulation;
	std::vector<int64_t> fitness(popSize);
	std::vector<int64_t> sortedFitness;
	std::vector<int> tmp;
	
	// init basic ordering (using relevant groups)
	//for (int x = 0; x < data.size(); x++)
	for (int x = data.size()-1; x >= 0; x--)
	{
		// init dynamic ordering
		if (!data[x].unused)
			readOrder.push_back(x);
		
		// init load/save data
		structure[x].whenLoad = data.size();
		structure[x].whenWrite = 0;
		structure[x].tmpValue = structure[x].numNeighbors;
	}
	
	// create initial random population
	for (int x = 0; x < popSize; x++)
	{
		currPopulation[x] = readOrder;
		int maxindex;
		EvalOrdering(data, structure, currPopulation[x], fitness[x], maxindex);

		// shuffle read order
		//std::random_shuffle ( readOrder.begin(), readOrder.end() );
		for (int m = 0; m < mutations; m++)
		{
			int a,b;
			do {
				a = random()%readOrder.size();
				b = random()%readOrder.size();
			} while (a == b);
			std::swap(readOrder[a], readOrder[b]);
		}
	}
	sortedFitness = fitness;
	std::sort(sortedFitness.begin(), sortedFitness.end());
	printf("Iteration %d; best fitness %llu (%1.2fGB)\n", 0, sortedFitness[0], sortedFitness[0]/8.0/1024.0/1024.0/1024.0);

	for (int iteration = 0; iteration < iterations; iteration++)
	{
		nextPopulation.resize(0);

		int cutoff = bestCutoff;
		while (sortedFitness[cutoff] == sortedFitness[cutoff+1])
		{
			cutoff--;
			if (cutoff == 0)
				break;
		}
		if (cutoff > 1)
		{
			for (int x = 0; x < fitness.size(); x++)
			{
				if (fitness[x] <= sortedFitness[cutoff])
				{
					nextPopulation.push_back(currPopulation[x]);
				}
			}
		}
		else {
			for (int x = 0; x < bestCutoff; x++)
			{
				nextPopulation.push_back(currPopulation[random()%popSize]);
			}
		}
		cutoff = nextPopulation.size();
		
		while (nextPopulation.size() < popSize)
		{
			int a, b;
			do {
				a = random()%cutoff;
				b = random()%cutoff;
			} while (a == b);
			
			// breed nextPopulation
			tmp = nextPopulation[a];
			tmp.resize(tmp.size()/2);
			for (int x = 0; x < nextPopulation[b].size(); x++)
			{
				if (std::find(tmp.begin(), tmp.end(), nextPopulation[b][x]) == tmp.end())
					tmp.push_back(nextPopulation[b][x]);
			}
			for (int x = 0; x < mutations; x++)
			{
				do {
					a = random()%tmp.size();
					b = random()%tmp.size();
				} while (a == b);
				std::swap(tmp[a], tmp[b]);
			}
			nextPopulation.push_back(tmp);
		}
		nextPopulation.swap(currPopulation);

		for (int x = 0; x < popSize; x++)
		{
			int maxindex;
			EvalOrdering(data, structure, currPopulation[x], fitness[x], maxindex);
		}
		sortedFitness = fitness;
		std::sort(sortedFitness.begin(), sortedFitness.end());
		printf("Iteration %d; best fitness %llu (%1.2fGB)\n", iteration, sortedFitness[0], sortedFitness[0]/8.0/1024.0/1024.0/1024.0);
	}

	for (int x = 0; x < currPopulation.size(); x++)
	{
		if (fitness[x] == sortedFitness[0])
		{
			readOrder = currPopulation[x];
			return;
		}
	}
	assert(!"Didn't find best population!");
}

void LocalSearch(const std::vector<bucketInfo> &data, std::vector<bucketStructure> &structure,
				 std::vector<int> &readOrder, int count)
{
	for (int x = 0; x < data.size(); x++)
	{
		// init dynamic ordering
		if (!data[x].unused)
			readOrder.push_back(x);
		
		// init load/save data
		structure[x].whenLoad = data.size();
		structure[x].whenWrite = 0;
		structure[x].tmpValue = structure[x].numNeighbors;
	}
	
	int64_t maxsum = 0, lastsum=0;
	int maxindex = 0;
	EvalOrdering(data, structure, readOrder, lastsum, maxindex);
	printf("Initial: Reached maximum memory usage of %1.2fGB (%1.2fMB) at step %d\n", lastsum/1024.0/1024.0/1024.0, lastsum/1024.0/1024.0, maxindex);
	
	srandom(81);
	// NO sort // 54.65GB [600k]
	
	// different sort
	// TODO: removed 12/17/14
	int current = 0;
	for (int x = 0; x < structure.size(); x++)
		//for (int x = structure.size()-1; x >= 0; x--)
	{
		if (data[x].unused)
			continue;
		for (int y = 0; y < structure[x].touchedGroupsID.size(); y++)
		{
			std::vector<int>::iterator iter = std::find(readOrder.begin()+current, readOrder.end(), structure[x].touchedGroupsID[y]);
			if (iter != readOrder.end())
			{
				std::swap(readOrder[current], *iter);
				current++;
			}
		}
	}
	EvalOrdering(data, structure, readOrder, lastsum, maxindex);
	printf("New Ordering: Reached maximum memory usage of %1.2fGB (%1.2fMB) at step %d\n", lastsum/1024.0/1024.0/1024.0, lastsum/1024.0/1024.0, maxindex);
	
	// ATTEMPT #1 sort by some order
	//	for (int x = 0; x < readOrder.size(); x++)
	//	{
	//		for (int y = x+1; y < readOrder.size(); y++)
	//		{
	//			//if (structure[readOrder[y]].lastGroup < structure[readOrder[x]].lastGroup) //53.14GB [600k]
	//			if (structure[readOrder[y]].lastGroup > structure[readOrder[x]].lastGroup) //52.58GB [600k]
	//			//if (structure[readOrder[y]].firstGroup > structure[readOrder[x]].firstGroup) //54.76GB [600k]
	//			//if (structure[readOrder[y]].firstGroup < structure[readOrder[x]].firstGroup) //52.85GB [600k]
	//			{
	//				int tmp = readOrder[x];
	//				readOrder[x] = readOrder[y];
	//				readOrder[y] = tmp;
	//			}
	//		}
	//	}
	
	// RANDOM!?! // 79.91GB [600k]
	//	std::random_shuffle ( readOrder.begin(), readOrder.end() );
	//	EvalOrdering(data, structure, readOrder, lastsum, maxindex, true);
	//	printf("Random: Reached maximum memory usage of %1.2fGB (%1.2fMB) at step %d\n", lastsum/1024.0/1024.0/1024.0/8.0, lastsum/1024.0/1024.0/8.0, maxindex);
	
	// ATTEMPT #4 local search
	
	
	for (int x = 0; x < count; x++) // 60k
	//for (int x = 0; x < 600000; x++)
	{
		int swapIndex, oldSwap;
		int modVal = 60;
		//		if (modVal > readOrder.size())
		//			modVal = readOrder.size();
		//
		//		bool changed = false;
		//		for (int off = -modVal; off <= modVal; off++)
		//		{
		//			if (maxindex+off >= 0 && maxindex+off < readOrder.size())
		//			{
		//				int swap1 = maxindex;
		//				int swap2 = maxindex+off;
		//				std::swap(readOrder[swap1], readOrder[swap2]);
		//				EvalOrdering(data, structure, readOrder, maxsum, maxindex);
		//
		//				if (maxsum < lastsum) {
		//					printf("[%4d] Reached maximum memory usage of %1.2fGB (%1.2fMB) at step %d\n", x, maxsum/1024.0/1024.0/1024.0/8.0, maxsum/1024.0/1024.0/8.0, maxindex);
		//					lastsum = maxsum;
		//					EvalOrdering(data, structure, readOrder, maxsum, maxindex, true);
		//					changed = true;
		//					break;
		//				}
		//				else {
		//					std::swap(readOrder[swap1], readOrder[swap2]);
		//				}
		//			}
		//		}
		//		if (!changed)
		//		{
		//			do {
		//				if (random()%2)
		//					maxindex = maxindex+random()%modVal+1;
		//				else
		//					maxindex = maxindex-random()%modVal-1;
		//			} while (maxindex < 0 || maxindex >= readOrder.size());
		//		}
		
		do {
			if (random()%2)
				swapIndex = maxindex+random()%modVal+1;
			else
				swapIndex = maxindex-random()%modVal-1;
		} while (swapIndex < 0 || swapIndex >= readOrder.size());
		
		if (x < 3*count/4)//50000)//200000)
		{
			do {
				if (random()%2)
					maxindex = maxindex+random()%5+1;
				else
					maxindex = maxindex-random()%5-1;
			} while (maxindex < 0 || maxindex >= readOrder.size());
		}
		
		std::swap(readOrder[maxindex], readOrder[swapIndex]);
		oldSwap = maxindex;
		
		EvalOrdering(data, structure, readOrder, maxsum, maxindex);
		//		// undo swap
		if (maxsum > lastsum)
		{
			std::swap(readOrder[oldSwap], readOrder[swapIndex]);
		}
		else if (maxsum < lastsum) {
			printf("[%4d] Reached maximum memory usage of %1.2fGB (%1.2fMB) at step %d\n", x, maxsum/1024.0/1024.0/1024.0, maxsum/1024.0/1024.0, maxindex);
			lastsum = maxsum;
			EvalOrdering(data, structure, readOrder, maxsum, maxindex);
		}
	}
}

//void ExhaustiveSearch(const std::vector<bucketInfo> &data, std::vector<bucketStructure> &structure,
//					  std::vector<int> &readOrder, int64_t &maxMemory, int &maxIndex)
//{
//	
//}

void EvalOrdering(const std::vector<bucketInfo> &data, std::vector<bucketStructure> &structure,
				  const std::vector<int> &readOrder, int64_t &maxMemory, int &maxIndex, bool print)
{
	Memory m;
	for (int x = 0; x < structure.size(); x++)
	{
		structure[x].tmpValue = structure[x].numNeighbors;
		assert(structure[x].touchedGroupsID.size() == structure[x].numNeighbors);
	}
	int64_t memUsage = 0, maxMemUsage = 0;
	int maxID;
	for (int x = 0; x < readOrder.size(); x++)
	{
		for (int y = 0; y < structure[readOrder[x]].touchedGroupsID.size(); y++)
		{
			// tmp is the number of neighbors yet to be written.
			// So, this means we haven't touched this group yet.
			if (structure[structure[readOrder[x]].touchedGroupsID[y]].tmpValue ==
				structure[structure[readOrder[x]].touchedGroupsID[y]].numNeighbors)
			{
				uint64_t entries = data[structure[readOrder[x]].touchedGroupsID[y]].numEntries;
				memUsage += entries;
				structure[structure[readOrder[x]].touchedGroupsID[y]].whenLoad = x;
				size_t add = m.Alloc((entries+7)/8);
				structure[structure[readOrder[x]].touchedGroupsID[y]].memoryAddress = add;
//				if (print)
//					printf("Group %d GETS memory address %lu\n", structure[readOrder[x]].touchedGroupsID[y], add);
				// TODO: revert this code if we want to optimize total entries not memory
				memUsage = m.GetCurrMemory()*8;
			}
		}
		if (memUsage > maxMemUsage)
		{
			maxID = x;
			maxMemUsage = memUsage;
			//m.Print();
		}
		for (int y = 0; y < structure[readOrder[x]].touchedGroupsID.size(); y++)
		{
			structure[structure[readOrder[x]].touchedGroupsID[y]].tmpValue--;
			if (structure[structure[readOrder[x]].touchedGroupsID[y]].tmpValue == 0)
			{
				memUsage -= data[structure[readOrder[x]].touchedGroupsID[y]].numEntries;
				structure[structure[readOrder[x]].touchedGroupsID[y]].whenWrite = x;
				m.Free(structure[structure[readOrder[x]].touchedGroupsID[y]].memoryAddress);
//				if (print)
//					printf("Group %d FREE memory address %lu\n", structure[readOrder[x]].touchedGroupsID[y],
//						   structure[structure[readOrder[x]].touchedGroupsID[y]].memoryAddress);

				// TODO: revert this code if we want to optimize total entries not memory
				memUsage = m.GetCurrMemory()*8;
			}
		}
	}
	maxMemory = maxMemUsage;
	maxIndex = maxID;
	
	//TODO:  Measure real memory usage based on VM allocation
	if (print)
	{
		printf("-->VM scheme requires %lu bytes (%1.2f MB / %1.2f GB) max usage\n",
			   m.GetMaxMemory(), m.GetMaxMemory()/1024.0/1024.0, m.GetMaxMemory()/1024.0/1024.0/1024.0);
	}
	maxMemory = m.GetMaxMemory();
	//m.Print();
}


//	// ATTEMPT #3
//	// add biggest remaining, and then add all related groups
//	for (int x = 0; x < readOrder.size(); ) // no increment -- it happsn below
//	{
//		int whichGroup = readOrder[x];
//		// find biggest outstanding group
//		for (int y = 0; y < structure.size(); y++)
//		{
//			if (structure[y].tmpValue != structure[y].numNeighbors && structure[y].tmpValue != 0)
//			{
//
//				if (!data[y].unused && data[y].numEntries > data[whichGroup].numEntries)
//				{
//					whichGroup = y;
//				}
//			}
//		}
//		if (data[whichGroup].numEntries < 1024*1024*1024)
//		{
//			for (int y = 0; y < structure.size(); y++)
//			{
//				if (structure[y].tmpValue != 0)
//				{
//
//					if (!data[y].unused && data[y].numEntries > data[whichGroup].numEntries)
//					{
//						whichGroup = y;
//					}
//				}
//			}
//		}
//		printf("Group %d has %llu entries remaining (max)\n", whichGroup, data[whichGroup].numEntries);
//		for (int y = x; y < readOrder.size(); y++)
//		{
//			if (structure[readOrder[y]].touchedGroups[whichGroup])
//			{
//				printf("Also adding group %d\n", readOrder[y]);
//				for (int z = 0; z < structure[readOrder[y]].touchedGroups.size(); z++)
//				{
//					if (structure[readOrder[y]].touchedGroups[z])
//					{
//						structure[z].tmpValue--;
//					}
//				}
//				int tmp = readOrder[x];
//				readOrder[x] = readOrder[y];
//				readOrder[y] = tmp;
//				x++;
//			}
//		}
//		structure[whichGroup].tmpValue = 0;
//	}

// ATTEMPT #1 sort by some order
//	for (int x = 0; x < readOrder.size(); x++)
//	{
//		for (int y = x+1; y < readOrder.size(); y++)
//		{
//			if ((structure[readOrder[y]].lastGroup < structure[readOrder[x]].lastGroup) ||
//				((structure[readOrder[y]].lastGroup == structure[readOrder[x]].lastGroup) &&
//				 (structure[readOrder[y]].firstGroup < structure[readOrder[x]].firstGroup)))
//			{
//				int tmp = readOrder[x];
//				readOrder[x] = readOrder[y];
//				readOrder[y] = tmp;
//			}
//		}
//	}

// ATTEMPT #2 add according to metric
//	for (int x = 0; x < readOrder.size(); x++)
//	{
//		int64_t targetMemory = 40ll*1024ll*1024ll*1024ll*8ll; // 40 GB
//		int64_t bestMemory = cc.getMaxSinglePlayerRank();// //readOrder.size();
//		int bestIndex = 0;
//
//		// choose the best group to add next
//
//		// maximize finished memory and minimize in-memory
//		for (int y = x; y < readOrder.size(); y++)
//		{
//			int64_t inMemory = 0;
//			int64_t finishedMemory = 0;
//			for (int z = 0; z < structure[readOrder[y]].touchedGroups.size(); z++)
//			{
//				if (structure[readOrder[y]].touchedGroups[z])
//				{
//					inMemory += data[z].numEntries;
//				}
//
//				if (structure[z].numNeighbors != structure[z].tmpValue)// && structure[z].tmpValue != 0)
//				{
//					finishedMemory += ((structure[z].numNeighbors - structure[z].tmpValue)*data[z].numEntries)/structure[z].numNeighbors;
//				}
//			}
//			//printf("--%d uses %lld; abs(%lld-%lld) = %lld\n", readOrder[y], metric, targetMemory, metric, abs(metric-targetMemory));
//			//metric = abs(metric-targetMemory);
//			int64_t metric = inMemory-finishedMemory;
//			if (inMemory > targetMemory)
//				metric += (inMemory-targetMemory)*1000;
//			if (inMemory < targetMemory/2)
//				metric += (targetMemory-inMemory);
//
//			if (metric < bestMemory)
//			{
//				bestIndex = y;
//				bestMemory = metric;
//			}
//		}
//		int tmp = readOrder[x];
//		readOrder[x] = readOrder[bestIndex];
//		readOrder[bestIndex] = tmp;
//
//		tmp = readOrder[x];
//		printf("[%d] %d next; score of %llu [%1.2fGB]\n", x, tmp, bestMemory, bestMemory/1024.0/1024.0/1024.0/8.0);
//		for (int z = 0; z < structure[tmp].touchedGroups.size(); z++)
//		{
//			if (structure[tmp].touchedGroups[z])
//			{
//				structure[z].tmpValue--;
//				printf("Entry %d hit by [%d], reduced to %d\n", z, tmp, structure[z].tmpValue);
//				if (structure[z].tmpValue < 0)
//				{
//					printf("Oh No!!!\n");
//					exit(0);
//				}
//			}
//		}
//	}
//
//	for (int z = 0; z < structure.size(); z++)
//	{
//		if (data[z].unused)
//			continue;
//
//		if (structure[z].tmpValue != 0)
//		{
//			printf("Entry %d not fully used !!!\n", z);
//			exit(0);
//		}
//	}

void InitBuckets(uint64_t maxBucketSize, std::vector<bucketChanges> &twoPieceChanges,
				 std::vector<bucketData> &buckets, int openSize, bool SYMMETRY, bool extraCoarseData)
{
	CCheckers cc;
	CCState s;
	
	int64_t maxVal2 = cc.getMaxSinglePlayerRank2();
	int64_t maxRank = cc.getMaxSinglePlayerRank();
	
	twoPieceChanges.resize(maxVal2);
	
	//	uint64_t bucketSize = 0;
	buckets.resize(1);
	buckets[0].theSize = 0;
	
	int currBucket = 0;
	
	// build data sets
	for (int64_t x = 0; x < maxRank; x++)
	{
		cc.unrankPlayer(x, s, 0);
		
		int64_t r1, r2, r3, r4;
		cc.rankPlayer(s, 0, r1, r2);
		cc.rankPlayerFirstTwo(s, 0, r3);
		cc.rankPlayerRemaining(s, 0, r4);
		assert(r1 == r3);
		assert(r2 == r4);
		
		int64_t maxValOther = cc.getMaxSinglePlayerRank2(r1);
		x+=maxValOther-1;
		
		if (SYMMETRY && cc.symmetricStart(s))
		{
		}
		else {
			if (buckets[currBucket].theSize > maxBucketSize)
			{
				currBucket++;
				buckets.resize(buckets.size()+1);
				buckets[currBucket].theSize = 0;
			}
			twoPieceChanges[r1].remainingEntries = maxValOther;
			twoPieceChanges[r1].lastDepthWritten = -1;
			twoPieceChanges[r1].currDepthWritten = -1;
			twoPieceChanges[r1].updated = false;
			twoPieceChanges[r1].changes.resize((maxValOther+openSize-1)/openSize);
			if (extraCoarseData)
			{
				twoPieceChanges[r1].coarseClosed.resize((maxValOther+openSize-1)/openSize);
				twoPieceChanges[r1].roundChanges.resize((maxValOther+openSize-1)/openSize);
			}
			twoPieceChanges[r1].nextChanges.resize((maxValOther+openSize-1)/openSize);
			buckets[currBucket].theSize += maxValOther;
			// 64-bit align data in buckets
			while (buckets[currBucket].theSize%openSize != 0)
				buckets[currBucket].theSize++;
			//numStatesLeft += maxValOther;
		}
	}
}

std::string SVGGetRGB(float r, float g, float b)
{
	std::string s;
	s = "rgb(";
	s += std::to_string(int(r*255)) + "," + std::to_string(int(g*255)) + "," + std::to_string(int(b*255));
	s += ")";
	return s;
}

//void GetXY(CCheckers &cc, int loc, float width, float spacing, float &x, float &y)
//{
//	int xi, yi;
//	cc.toxy_fast(loc, xi, yi);
//
//	y = spacing+yi*spacing;
//	x = width/2+spacing/2.0+xi*spacing-spacing*cc.count[yi]/2.0;
//}

void MakeSVG(const CCState &s, const char *output)
{
	CCheckers cc;
	CCState tmp;
	cc.Reset(tmp);
	float height = 400;
	float width = 400;
	std::string str;
	str = "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width = \""+std::to_string(width)+"\" height = \""+std::to_string(height)+"\" ";
	str += "viewBox = \"0 0 "+std::to_string(width)+" "+std::to_string(height)+"\" ";
	str += " preserveAspectRatio = \"none\"";
	str += " shape-rendering=\"auto\""; // crispEdges
	str += ">\n";

	// White background rectangle
	str += "<rect x=\"" + 0;
	str += "\" y=\"" + 0;
	str += "\" width=\""+std::to_string(width)+"\" height=\""+std::to_string(height)+"\" ";
	str += " style=\"fill:"+SVGGetRGB(1,1,1)+"\" />\n";
	
	float spacing = height/(ROWS+1.0f);
	float pieceSize = 1.25*spacing/sqrt(3.0);
	float centerLine = width/2.0f;
	printf("Space between rows: %1.2f\n", spacing);
	
	// Draw board
	int cnt = 0;
	int goals[NUM_SPOTS] = {0};
	
	for (int y = 0; y < NUM_PLAYERS; y++)
	{
		for (int x = 0; x < NUM_PIECES; x++)
		{
			goals[cc.places[1-y][x]] = y+1;
		}
	}
	
	for (int y = 0; y < ROWS; y++)
	{
		for (int x = 0; x < cc.count[y]; x++)
		{
			float yloc = spacing+y*spacing;
			float xloc = width/2+spacing/2.0+x*spacing-spacing*cc.count[y]/2.0;
			printf("Drawing at (%f, %f)\n", xloc, yloc);
			str += "<circle cx=\"" + std::to_string(xloc);
			str += "\" cy=\"" + std::to_string(yloc);
			str += "\" r=\""+std::to_string(pieceSize/2);
			if (s.board[cnt] == 0)
				str += "\" style=\"fill:none;stroke:";
			if (s.board[cnt] == 1)
				str += "\" style=\"fill:"+SVGGetRGB(1, 0, 0)+";stroke:";
			if (s.board[cnt] == 2)
				str += "\" style=\"fill:"+SVGGetRGB(0, 0, 1)+";stroke:";

			switch (goals[cnt])
			{
				case 0:
					str += "black";
					str += ";stroke-width:"+std::to_string(0.05f*pieceSize)+"\" />\n";
					break;
				case 1:
					str += "red";
					str += ";stroke-width:"+std::to_string(0.1f*pieceSize)+"\" />\n";
					break;
				case 2:
					str += "blue";
					str += ";stroke-width:"+std::to_string(0.1f*pieceSize)+"\" />\n";
					break;
				default:
					str += "yellow";
					str += ";stroke-width:"+std::to_string(0.05f*pieceSize)+"\" />\n";
					break;
			}
			cnt++;
		}
	}

	str += "<circle cx=\"" + std::to_string(spacing);
	str += "\" cy=\"" + std::to_string(spacing);
	str += "\" r=\""+std::to_string(pieceSize/2);
	if (s.toMove == 0)
		str += "\" style=\"fill:"+SVGGetRGB(1, 0, 0)+";stroke:black;";
	if (s.toMove == 1)
		str += "\" style=\"fill:"+SVGGetRGB(0, 0, 1)+";stroke:black;";
	if (s.toMove == 2)
		str += "\" style=\"fill:"+SVGGetRGB(0, 0, 1)+";stroke:black;";
	str += ";stroke-width:"+std::to_string(0.05f*pieceSize)+"\" />";
	
	str += "\n<text x=\""+std::to_string(spacing+3*pieceSize/4)+"\" y=\""+std::to_string(spacing)+"\" fill=\"black\" alignment-baseline=\"middle\" font-family=\"Impact\" font-size=\""+std::to_string(pieceSize)+"px\">Next</text>\n";

//	for (int x = 0; x < NUM_SPOTS; x++)
//	{
//		float xloc, yloc;
//		GetXY(cc, x, width, spacing, xloc, yloc);
//		str += "<circle cx=\"" + std::to_string(xloc);
//		str += "\" cy=\"" + std::to_string(yloc);
//		str += "\" r=\""+std::to_string(pieceSize/2);
//		str += "\" style=\"fill:none;stroke:"+SVGGetRGB(0, 0, 0);
//		str += ";stroke-width:"+std::to_string(0.05f*pieceSize)+"\" />";
//	}

	// Draw pieces
//	str += "<circle cx=\"" + to_string_with_precision(x);
//	str += "\" cy=\"" + to_string_with_precision(y);
//	str += "\" r=\""+to_string_with_precision(radius)+"\" style=\"fill:"+SVGGetRGB(c)+";stroke-width:1\" />";

	
//	s += "<!--";
//	s += comment;
//	s += "-->\n";
	str += "</svg>";

	
	std::fstream myFile;
	
	std::fstream svgFile;
	svgFile.open(output, std::fstream::out | std::fstream::trunc);
	svgFile << str;
	svgFile.close();

}

