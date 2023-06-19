/*
 *  screenUtil.cpp
 *  Solve Chinese Checkers
 *
 *  Created by Nathan Sturtevant on 6/20/11.
 *  Copyright 2011 University of Denver. All rights reserved.
 *
 */

#include "screenUtil.h"
#include <stdio.h>

void clrscr()
{
	printf("%c[H%c[J", 27, 27);
}

void topscr()
{
	printf("%c[H", 27);
}

void setcolor(int color, int mode)
{
	printf("%c[%d;%dm", 27, mode, color);
}

void gotoxy(int x, int y)
{
	printf("%c[%d;%df", 27, y, x);
}
