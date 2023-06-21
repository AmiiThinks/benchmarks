//
//  Driver.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/17/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include <unordered_map>
#include <set>
#include <deque>
#include <algorithm>
#include "CCUtils.h"
#include "Driver.h"
#include "Minimax.h"
#include "MinimaxDB.h"
#include "DistEval.h"
#include "LBDistEval.h"
#include "DistDBEval.h"
#include "DBEval.h"
#include "TDRegression.h"
#include "TDRegression2.h"
#include "UCB.h"
#include "UCT.h"
#include "CCEndgameData.h"
#include "DBPlusEval.h"
#include "UCTDB.h"

template <typename eval = TDEval, typename train = TDRegression>
eval Train(int numIterations = 50000);

CCMove *GetRandomMove(CCheckers *cc, CCState &s)
{
    CCMove *m = cc->getMoves(s);
    int count = m->length();
    int which = random()%count;
    while (which != 0)
    {
        which--;
        CCMove *tmp = m;
        m = m->next;
        tmp->next = 0;
        cc->freeMove(tmp);
    }
    cc->freeMove(m->next);
    m->next = 0;
    return m;
}

CCMove *GetRandomForwardMove(CCheckers *cc, CCState &s)
{
    CCMove *m = cc->getMovesForward(s);
    if (m == 0)
        m = cc->getMoves(s);
    int count = m->length();
    int which = random()%count;
    while (which != 0)
    {
        which--;
        CCMove *tmp = m;
        m = m->next;
        tmp->next = 0;
        cc->freeMove(tmp);
    }
    cc->freeMove(m->next);
    m->next = 0;
    return m;
}


void GetRandomState(CCheckers *cc, CCState &s, int walkLen = 50)
{
    for (int x = 0; x < walkLen; x++)
    {
        CCMove *m = GetRandomMove(cc, s);
        cc->ApplyMove(s, m);
        cc->freeMove(m);
    }
}

void PlayGame(Player *p1, Player *p2, double timeLimit = 1.0)
{
    bool drawByRepetition = false;
    std::vector<Player *> p;
    std::vector<CCMove *> moves;
    p.push_back(p1);
    p.push_back(p2);
    std::vector<CCState> states;
    CCheckers cc;
    CCState s;
    cc.Reset(s);
    p1->Reset();
    p2->Reset();
    while (!cc.Done(s) && !drawByRepetition)
    {
        states.push_back(s);
        double value;
        s.Print();
        CCMove *next = p[s.toMove]->GetNextAction(&cc, s, value, timeLimit);
        cc.ApplyMove(s, next);
        moves.push_back(next);
        for (auto &st : states)
        {
            if (st == s)
            {
                drawByRepetition = true;
                break;
            }
        }
    }
    s.Print();
    if (drawByRepetition)
    {
        std::cerr << "Loss by repetition\n";
        std::cerr << p[s.toMove]->GetName() << " won\n";
        std::cerr << p[1-s.toMove]->GetName() << " lost\n";
    }
    else {
        std::cerr << p[cc.Winner(s)]->GetName() << " won\n";
        std::cerr << p[1-cc.Winner(s)]->GetName() << " lost\n";
    }
    while (!moves.empty())
    {
        cc.freeMove(moves.back());
        moves.pop_back();
    }
}

void RegularGame()
{
    DistEval d;
    DistDBEval db(dataPrefix, 1962, 36);
    Minimax<DistEval> p1(&d, false);
    Minimax<DistDBEval> p2(&db, false);
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&p2, &p1, 1);
        PlayGame(&p1, &p2, 1);
    }
}

void DBGame()
{
    //DistDBEval db1(dataPrefix, 1962, 36);
    //DistDBEval db2(dataPrefix, 975, 15);
    //DistDBEval db2(dataPrefix, 1638, 28);
    DistEval d;
    //DistDBEval db(dataPrefix, 975, 15);
    DistDBEval db(dataPrefix, 975, 15);
    //DistDBEval db(dataPrefix, 1962, 36);
    //DistDBEval db2(dataPrefix, 1638, 28);
// 1638
    Minimax<DistEval> p1(&d, false);
    Minimax<DistDBEval> p2(&db, false);
    // 1638, 28
    // 1962, 36
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&p2, &p1);
        PlayGame(&p1, &p2);
    }
}


void UCBvMinimax()
{
    DistEval d;
    Minimax<DistEval> p1(&d, false);
    UCB u;
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&u, &p1, 5);
        PlayGame(&p1, &u, 5);
    }
}

void UCTvMinimax()
{
    DistDBEval db1(dataPrefix, 975, 15);
    DistDBEval db2(dataPrefix, 1962, 36);
    DistEval d;
    //RandomPlayout r;
    BestPlayout b;
    //BackPlayout back;
    Minimax<DistDBEval> p1(&db2, false);
    UCT<BestPlayout, DistDBEval> u(10, &b, &db1);

    for (int x = 0; x < 100; x++)
    {
        PlayGame(&u, &p1, 1);
        PlayGame(&p1, &u, 1);
    }
}

// back is better for DB eval, but not for dist eval
void UCTvUCT()
{
    //DBEval db2(dataPrefix, 975, 15);
    DistDBEval db1(dataPrefix, 975, 15);

    db1.lookup = kDBXorDist;
    //db2.lookup = kDBXorDist;

    DistEval d;
    BestPlayout b;
    //BackPlayout back;
    
//  UCT<BestPlayout, DistDBEval> p1(10, &b, &d);
    UCT<BestPlayout, DistEval> p1(10, &b, &d);
//    TDEval e = Train();
//    UCT<BestPlayout, TDEval> p2(10, &b, &e);
    UCT<BestPlayout, DistEval> p2(10, &b, &d);
    //p1.initFromDB = false;
    //p2.initFromDB = false;
    //p1.playOutDepth = 10;
    //p2.playOutDepth = 5;

    for (int x = 0; x < 1; x++)
    {
        PlayGame(&p2, &p1, 1);
        PlayGame(&p1, &p2, 1);
    }
}

template <typename eval, typename train>
eval Train(int numIterations)
{
    train tdr(.25, 0.8);
    //train tdr(.05, 0.8);
    std::vector<Player *> p;
    std::vector<CCMove *> moves;
    p.push_back(&tdr);
    p.push_back(&tdr);
    std::cout << "Training " << tdr.GetName() << "\n";
    CCheckers cc;
    CCState s;
    for (int x = 0; x < numIterations; x++)
    {
        cc.Reset(s);
        std::swap(p[0], p[1]);
        while (!cc.Done(s))
        {
            double value;
            //s.Print();
            CCMove *next = p[s.toMove]->GetNextAction(&cc, s, value, 0.05);
            cc.ApplyMove(s, next);
            moves.push_back(next);
        }
        //s.Print();
        //std::cerr << p[cc.Winner(s)]->GetName() << " won\n";
        //std::cerr << p[1-cc.Winner(s)]->GetName() << " lost\n";
        tdr.Train(&cc, moves);
        
        while (!moves.empty())
        {
            cc.freeMove(moves.back());
            moves.pop_back();
        }
        if (0 == x%10000)
        {
//          tdr.Print();
            printf("Training round %d done\n", x);
            fflush(stdout);
            //          DistEval d;
//          TDEval e = tdr.GetTDEval();
//          Minimax<TDEval> p1(&e, false);
//          Minimax<DistEval> p2(&d, false);
//          PlayGame(&p1, &p2, 10);
        }
    }
    
    tdr.Print();
    return tdr.GetTDEval();
}

void MinimaxTrainVersusUCT()
{
    DistEval d;
    BestPlayout b;
    TDEval e = Train();
    Minimax<DistEval> p1(&d, false);
    UCT<BestPlayout, TDEval> p2(10, &b, &e);
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}

void MinimaxDistVersusTrain()
{
    DistEval d;
    TDEval e = Train();
    Minimax<TDEval> p1(&e, false);
    Minimax<DistEval> p2(&d, false);
    for (int x = 0; x < 100; x++)
    {   
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}

void MinimaxTrain1VersusTrain2()
{
    TDEval2 e2 = Train<TDEval2, TDRegression2>(100000);
    TDEval e1 = Train();
    Minimax<TDEval> p1(&e1, false);
    Minimax<TDEval2> p2(&e2, false);
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}


void MinimaxVersusZW()
{
    TDEval e = Train();
//  DistEval d;
    Minimax<TDEval> p1(&e, false);
    Minimax<TDEval> p2(&e, false);
    p1.zeroWindow = true;
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}


void MinimaxDBPlusTrain()
{
    TDEval e = Train();
    DBPlusEval<TDEval> db(&e, dataPrefix, 1962, 36);
    Minimax<TDEval> p1(&e, false);
    Minimax<DBPlusEval<TDEval>> p2(&db, false);
    for (int x = 0; x < 100; x++)
    {   
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}

void MinimaxDBDistVersusDBTrain()
{
    DistEval d;
    TDEval e = Train();
    DBPlusEval<DistEval> dbdist(&d, dataPrefix, 1962, 36);
    DBPlusEval<TDEval> dbtd(&e, dataPrefix, 1962, 36);
    Minimax<DBPlusEval<DistEval>> p1(&dbdist, false);
    Minimax<DBPlusEval<TDEval>> p2(&dbtd, false);
    for (int x = 0; x < 100; x++)
    {   
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}


void TestHH(int numTests = 100)
{
    CCheckers cc;
    CCState s;
    DistEval d;
    Minimax<DistEval> p2(&d, false);
    for (int x = 0; x < numTests; x++)
    {
        cc.Reset(s);
        GetRandomState(&cc, s);
        double v1, v2;
        p2.transpositionTable = false;
        p2.historyHeuristic = false;
        cc.freeMove(p2.GetNextAction(&cc, s, v1, 10000, 7));
        uint64_t nodes = p2.GetNodesExpanded();
        p2.historyHeuristic = true;
        cc.freeMove(p2.GetNextAction(&cc, s, v2, 10000, 7));
        printf("Round %d summary: %llu without HH, %llu with\n", x, nodes, p2.GetNodesExpanded());
        assert(v1 == v2);
    }
}

void TestTT(int numTests = 100)
{
    CCheckers cc;
    CCState s;
    DistEval d;
    Minimax<DistEval> p2(&d, false);
    p2.historyHeuristic = true;
    srandom(102);
    for (int x = 0; x < numTests; x++)
    {
        Timer t1, t2;
        cc.Reset(s);
        GetRandomState(&cc, s);
        double v1, v2;
        p2.transpositionTable = true;
        t1.StartTimer();
        cc.freeMove(p2.GetNextAction(&cc, s, v1, 10000, 7));
        t1.EndTimer();
        uint64_t nodes = p2.GetNodesExpanded();
        p2.transpositionTable = false;
        t2.StartTimer();
        cc.freeMove(p2.GetNextAction(&cc, s, v2, 10000, 7));
        t2.EndTimer();

        printf("Round %d summary: %llu/%1.2fs with TT, %llu/%1.2fs without (gain %1.2f/%1.2f)\n", x,
               nodes, t1.GetElapsedTime(),
               p2.GetNodesExpanded(), t2.GetElapsedTime(),
               float(p2.GetNodesExpanded())/nodes, t2.GetElapsedTime()/t1.GetElapsedTime());
        assert(v1 == v2);
    }
}

void TestFlip()
{
    CCheckers cc;
    CCState s;
    for (int x = 0; x < 10000; x++)
    {
        cc.Reset(s);
        GetRandomState(&cc, s);
        //s.Print();
        int64_t rank = cc.rankPlayerFlipped(s, 1);
        int64_t r1a, r2a;
        int64_t r1b, r2b;
        cc.rankPlayerFlipped(s, 1, r1a, r2a);
        cc.unrankPlayer(rank, s, 0);
        //s.Print();
        cc.rankPlayer(s, 0, r1b, r2b);
        //s.Print();
        assert(r1a == r1b);
        assert(r2a == r2b);
    }
}

void TestLookup()
{
    CCheckers cc;
    CCEndGameData db(dataPrefix, 1962);
    CCState s;
    srandom(104);
    int64_t r2Count = cc.getMaxSinglePlayerRank2();
    int64_t id = 0;
    int depth;
    for (int64_t x = 0; x < r2Count; x++)
    {
        if (x >= 1962)
        {
            cc.unrankPlayer(id, s, 0);
            bool valid = db.GetDepth(s, 0, depth);
            assert(valid);
            printf("2-piece %lld offset %lld distance %d\n", x, 0ll, depth);
            id += cc.getMaxSinglePlayerRank2(x)-1;
            
            cc.unrankPlayer(id, s, 0);
            valid = db.GetDepth(s, 0, depth);
            assert(valid);
            printf("2-piece %lld offset %lld distance %d\n", x, cc.getMaxSinglePlayerRank2(x)-1, depth);
            id++;
        }
        else {
            id += cc.getMaxSinglePlayerRank2(x);
        }
    }
    for (int x = 0; x < 0; x++)
    {
        cc.Reset(s);
        int cnt = 0;
        while ((GetBackPieceAdvancement(s, 0) < 36 ||
                GetBackPieceAdvancement(s, 1) < 36) && ++cnt < 10000)
        {
            CCMove *m = GetRandomForwardMove(&cc, s);
            cc.ApplyMove(s, m);
            cc.freeMove(m);
            //s.Print();
        }
        if (cnt < 10000)
        {
            //s.Print();
            int d1 = GetDepth(dataPrefix, s, 0);
            int d2 = GetDepth(dataPrefix, s, 1);
            printf("Player 1 %d steps away; Player 2 %d steps away\n", d1, d2);
            int d1a, d2a;
            assert(db.GetDepth(s, 0, d1a));
            assert(d1a == d1);
            assert(db.GetDepth(s, 1, d2a));
            assert(d2a == d2);
        }
    }
}

void Minimax15()
{
    DistEval d;
    
    //DistDBEval db(dataPrefix, 1962, 36);
    Minimax<DistEval> p1(&d, false);
    MinimaxDB::MinimaxDB<DistEval> p2(dataPrefix, 975, 15, &d, false);
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}

void UCTWithDepthFix()
{
    // doesn't play well
    DistEval d;
    BestPlayout b;
    DBPlusEval<DistEval> db1(&d, dataPrefix, 1638, 28);
    DBPlusEval<DistEval> db2(&d, dataPrefix, 1638, 28);
    UCT<BestPlayout, DBPlusEval<DistEval>> p1(10, &b, &db1);
    UCT<BestPlayout, DBPlusEval<DistEval>> p2(10, &b, &db2);
    db1.validateDBValue = true;
    db2.validateDBValue = false;
    for (int x = 0; x < 100; x++)
    {
        PlayGame(&p1, &p2, 1);
        PlayGame(&p2, &p1, 1);
    }
}

void UCTIncrementalDB()
{
//  BestPlayout b;
//  DistEval d;
//  UCTDB<BestPlayout, DistEval> p1(10, &b, &d);
}

int64_t rand64()
{
    uint64_t a = random();
    a<<=32;
    a|=random();
    return a;
}

void ExtractOptimalPath(CCheckers &cc, CCState &s, std::vector<int64_t> &path)
{
    path.push_back(cc.rankPlayer(s, 0));
    int parentDepth = GetDepth(fullDataPrefix, s, 0);
    CCMove *m = cc.getMoves(s);
    //s.Print();
    //std::cout << "Legal moves: ";
    //m->Print(1);
    for (CCMove *t = m; t; t = t->next)
    {
        cc.ApplyMove(s, t);
        int childDepth = GetDepth(fullDataPrefix, s, 0);
        if (childDepth == (parentDepth+14)%15)
        {
            s.toMove = 0;
            //s.Print();
            CCMove *tmp = t->clone(cc);
            cc.freeMove(m);
            m = 0;
            ExtractOptimalPath(cc, s, path);
            cc.UndoMove(s, tmp);
            cc.freeMove(tmp);
            break;
        }
        cc.UndoMove(s, t);
    }
    cc.freeMove(m);
}

void ExtractNearSeed(int64_t seed, int limit = 10)
{
    CCheckers cc;
    CCState s;
    std::unordered_map<int64_t, int> states;
    std::set<int64_t> seeds;
    std::deque<int64_t> queue;
    std::vector<int64_t> path;
    queue.push_back(seed);
    while (seeds.size() < limit)
    {
        int64_t next = queue.front();
        queue.pop_front();
        cc.unrankPlayer(next, s, 0);

        CCMove *m = cc.getMoves(s);
        for (CCMove *t = m; t; t = t->next)
        {
            cc.ApplyMove(s, t);
            int64_t rank = cc.rankPlayer(s, 0);
            if (seeds.find(rank) == seeds.end())
            {
                seeds.insert(rank);
                queue.push_back(rank);
            }
            cc.UndoMove(s, t);
        }
        cc.freeMove(m);
    }
    for (auto &next : seeds)
    {
        cc.unrankPlayer(next, s, 0);
        ExtractOptimalPath(cc, s, path);
        std::reverse(path.begin(), path.end());
        for (int y = 0; y < path.size(); y++)
        {
            states[path[y]] = y;
        }
        path.resize(0);
    }

    for (auto &x : states)
    {
        std::cout << x.first << " " << x.second << "\n";
    }
}

void ExtractDepths(int count = 1000000)
{
    CCheckers cc;
    CCState s;
    std::unordered_map<int64_t, int> states;
    
    int64_t maxVal = cc.getMaxSinglePlayerRank();
    std::vector<int64_t> path;
    for (int x = 0; x < count; x++)
    {
        int64_t next = rand64()%(maxVal/2);
        cc.unrankPlayer(next, s, 0);
        ExtractOptimalPath(cc, s, path);
        std::reverse(path.begin(), path.end());
        for (int y = 0; y < path.size(); y++)
        {
            states[path[y]] = y;
        }
        path.resize(0);
    }
    for (auto &x : states)
    {
        std::cout << x.first << " " << x.second << "\n";
    }
}

int GetLowerBound(const CCheckers &cc, const CCState &s, int start, int who)
{
    int inPlace = 0;
    int moves1 = 0;
    int common = 0;
    uint8_t dist[17];
    memset(dist, 0, 17);
    for (int x = 0; x < NUM_PIECES; x++)
    {
        int next = cc.distance(s.pieces[who][x], cc.getGoal(who));
        dist[next]++;
    }
    for (int x = 0; x < 17 && inPlace != NUM_PIECES; x++)
    {
        if (x < 4)
        {
            inPlace += dist[x];
        }
        else {
            if (dist[x] == 0)
            {
                // at least one move to fill in empty row for later jumping
                moves1++;
            }
            else {
                common += dist[x];
                inPlace += dist[x];
            }
        }
    }
    while (start < common+moves1)
        start += 15;
    assert(start <= 33);
    return start;
}

void GetFeatures(CCheckers &cc, CCState &s, std::vector<int> &features, int depth)
{
    features.clear();
    uint8_t dist[17];
    uint8_t gap[17];
    memset(dist, 0, 17);
    memset(gap, 0, 17);

    //- # of pieces in the row [0, 1, 2, 3, 4, 5+] {0-5}  \/
    //- pieces in later rows? {6}  \/
    //- Is there a gap of 2 spaces? [between] {7}
    //- Is there a gap of 3 spaces? [between] {8}
    //- Is there a gap of 4+ spaces? [between] {9}

    int lastRow = 0;
    for (int x = 0; x < NUM_PIECES; x++)
    {
        int next = cc.distance(s.pieces[0][x], cc.getGoal(0));
        dist[next]++;
        lastRow = std::max(lastRow, next);
        if (dist[next] > 1)
        {
            gap[next] = std::max(0, cc.distance(s.pieces[0][x], s.pieces[0][x-1]));
        }
    }
    for (int x = 0; x < 17; x++)
    {
        int num = dist[x];
        if (num > 5)
            num = 5;
        features.push_back(x*10+num);
        if (x <= lastRow)
            features.push_back(x*10+6);
        if (gap[x] > 1)
        {
            if (gap[x] > 4)
                gap[x] = 4;
            features.push_back(x*10+7+gap[x]-2);
        }
    }

//  for (int z = 0; z < NUM_PIECES; z++)
//  {
//      features.push_back(s.pieces[0][z]);
//      for (int w = 0; w < NUM_PIECES; w++)
//      {
//          features.push_back(81+34+s.pieces[0][z]*81+s.pieces[0][w]);
//      }
//  }
//  int lb = GetLowerBound(cc, s, depth%15, 0);
//  features.push_back(81+lb);
}

void TrainRegression(LinearRegression &r, const char *file, int numTrainingRounds = 500000)
{
    CCheckers cc;
    CCState s, flip;
    
    std::vector<std::vector<uint64_t> > data(34);
    std::vector<long> correct(34);
    
    printf("Loading and removing obvious states\n");
    FILE *f = fopen(file, "r+");
    while (!feof(f))
    {
        uint64_t rank;
        int depth;
        if (fscanf(f, "%llu %d", &rank, &depth) == 2)
        {
            assert(data.size() > depth);
            cc.unrankPlayer(rank, s, 0);
            int lb = GetLowerBound(cc, s, depth%15, 0);
            if (lb+15 <= 33) // has two or more possible values
            {
                data[depth].push_back(rank);
            }
        }
        else {
            break;
        }
    }
    fclose(f);
    //const int numTrainingRounds = 500000;
    std::vector<int> features;
    // train on data
    for (int x = 0; x < numTrainingRounds; x++)
    {
        if (0 == x%10000)
            printf("Training %d of %d\n", x, numTrainingRounds);
        for (int y = 0; y < data.size(); y++)
        {
            if (data[y].size() == 0)
                continue;
            uint64_t nextRank = data[y][random()%data[y].size()];
            cc.unrankPlayer(nextRank, s, 0);
            
            if (0 == random()%2)
            {
                cc.FlipPlayer(s, flip, 0);
                s = flip;
            }
            
            GetFeatures(cc, s, features, y);
            
            r.setRate(0.1/features.size());
            r.train(features, y);
        }
    }
}

void TestRegression(LinearRegression &r, const char *file)
{
    CCheckers cc;
    CCState s, flip;
    
    std::vector<std::vector<uint64_t> > data(34);
    std::vector<long> correct(34);
    
    FILE *f = fopen(file, "r+");
    while (!feof(f))
    {
        uint64_t rank;
        int depth;
        if (fscanf(f, "%llu %d", &rank, &depth) == 2)
        {
            assert(data.size() > depth);
            data[depth].push_back(rank);
        }
        else {
            break;
        }
    }
    fclose(f);
    std::vector<int> features;


    // Test on all data
    // Can we get within +- 6 of the correct answer?
    for (int y = 0; y < data.size(); y++)
    {
        for (int x = 0; x < data[y].size(); x++)
        {
            cc.unrankPlayer(data[y][x], s, 0);
            int lb = GetLowerBound(cc, s, y%15, 0);
            if (lb+15 > 33)
            {
                if (lb != y)
                {
                    s.Print();
                    printf("[%s] ERROR! lb: %d; y: %d\n", file, lb, y);
                    printf("State %llu\n", cc.rankPlayer(s, 0));
                    exit(0);
                }
                else {
                    correct[y]+=2;
                }
            }
            else {
                for (int flp = 0; flp < 2; flp++)
                {
                    uint64_t nextRank = data[y][x];
                    cc.unrankPlayer(nextRank, s, 0);
                    if (flp)
                    {
                        cc.FlipPlayer(s, flip, 0);
                        s = flip;
                    }
                    
                    GetFeatures(cc, s, features, y);
                    
                    double result = r.test(features);
                    int lb = GetLowerBound(cc, s, y%15, 0);
                    if (result < lb)
                        result = lb;
                    if (result >= y-6 && result <= y+6)
                    {
                        correct[y]++;
                    }
                    else {
                        s.Print();
                        printf("State %llu\n", cc.rankPlayer(s, 0));
                        printf("Data: %d; Predicted %1.2f; Lower Bound: %d; actual %d\n", y%15, result, GetLowerBound(cc, s, y%15, 0), y);
                    }
                }
            }
        }
    }
    
    int64_t right = 0;
    int64_t total = 0;
    for (int x = 0; x < data.size(); x++)
    {
        printf("Depth %2d: [%1.5f] %lu of %lu correct\n", x, double(correct[x])/(2*data[x].size()),correct[x], 2*data[x].size());
        right += correct[x];
        total += 2*data[x].size();
    }
    printf("Final: [%1.5f] %lld of %lld correct\n", double(right)/total, right, total);
}

void TrainDepthClassifier()
{
//  LinearRegression r(81+34+81*81, 0.01);
    LinearRegression r(170, 0.01);
    
    TrainRegression(r, "/Users/nathanst/Development/cc/cc-train-new1.txt", 500000);
//  TrainRegression(r, "/Users/nathanst/Development/cc/cc-train-full.txt");
//  TrainRegression(r, "/Users/nathanst/Development/cc/cc-train-new.txt");
    TestRegression(r, "/Users/nathanst/Development/cc/cc-train-full.txt");
    TestRegression(r, "/Users/nathanst/Development/cc/cc-train-new.txt");
}

int main(int argc, char **argv)
{
    //MinimaxTrain1VersusTrain2();
    //ExtractDepths();
    //TrainDepthClassifier();
    //UCTWithDepthFix();
    //Minimax15();
    UCTvUCT();
    //UCTvMinimax();
    //DBGame();
    //RegularGame();
    //TestHH();
    //TestTT();
    //TrainGame();
    //TestFlip();
    //TestLookup();

    //MinimaxVersusZW();
    //MinimaxDistVersusTrain();
    //MinimaxDBPlusTrain();
    //MinimaxDBDistVersusDBTrain();
    //MinimaxTrainVersusUCT();
    
    return 0;
}
